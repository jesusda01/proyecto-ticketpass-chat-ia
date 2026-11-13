import json
import os
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen

import boto3
from boto3.dynamodb.conditions import Attr, Key

from groq_client import chat_completion

CHAT_TABLE = os.getenv("DYNAMODB_TABLE", "chat-memory-dev")

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(CHAT_TABLE)


def _response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
            "Access-Control-Allow-Methods": "OPTIONS,GET,POST"
        },
        "body": json.dumps(body),
    }


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _sort_messages(messages):
    return sorted(messages, key=lambda x: x.get("sk", ""))


def create_chat(event, context):
    try:
        payload = json.loads(event.get("body") or "{}")
        title = (payload.get("title") or "Nuevo chat").strip()
        chat_id = payload.get("chatId") or f"chat-{int(time.time() * 1000)}"
        created_at = _now_iso()

        table.put_item(
            Item={
                "pk": f"CHAT#{chat_id}",
                "sk": "META",
                "chatId": chat_id,
                "title": title,
                "createdAt": created_at,
            }
        )

        return _response(201, {"chatId": chat_id, "title": title, "createdAt": created_at})
    except Exception as exc:
        return _response(500, {"error": "create_chat_failed", "detail": str(exc)})


def list_chats(event, context):
    try:
        result = table.scan(
            FilterExpression=Attr("sk").eq("META")
        )
        items = result.get("Items", [])
        chats = sorted(items, key=lambda x: x.get("createdAt", ""), reverse=True)
        return _response(200, {"chats": chats})
    except Exception as exc:
        return _response(500, {"error": "list_chats_failed", "detail": str(exc)})


def get_chat(event, context):
    try:
        chat_id = event.get("pathParameters", {}).get("chatId")
        if not chat_id:
            return _response(400, {"error": "chatId_required"})

        response = table.query(
            KeyConditionExpression=Key("pk").eq(f"CHAT#{chat_id}")
        )
        items = response.get("Items", [])
        meta = next((item for item in items if item.get("sk") == "META"), None)
        messages = [item for item in items if item.get("sk", "").startswith("MSG#")]
        messages = _sort_messages(messages)

        return _response(200, {"chat": meta, "messages": messages})
    except Exception as exc:
        return _response(500, {"error": "get_chat_failed", "detail": str(exc)})


def send_message(event, context):
    try:
        chat_id = event.get("pathParameters", {}).get("chatId")
        if not chat_id:
            return _response(400, {"error": "chatId_required"})

        payload = json.loads(event.get("body") or "{}")
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            return _response(400, {"error": "prompt_required"})

        history_response = table.query(
            KeyConditionExpression=Key("pk").eq(f"CHAT#{chat_id}")
        )
        history_items = history_response.get("Items", [])
        history_messages = [item for item in history_items if item.get("sk", "").startswith("MSG#")]
        history_messages = _sort_messages(history_messages)

        groq_messages = []
        for message in history_messages:
            groq_messages.append({"role": message.get("role"), "content": message.get("content")})

        groq_messages.append({"role": "user", "content": prompt})

        # Extraer estrictamente el texto plano del diccionario devuelto por Groq
        completion = chat_completion(groq_messages)
        if isinstance(completion, dict):
            answer = completion.get("content", "")
        else:
            answer = str(completion)

        # Interceptar si el bot devolvió el JSON completo con los datos del ticket
        try:
            if "{" in answer and "}" in answer and "evento" in answer and "zona" in answer and "cantidad" in answer:
                start_idx = answer.find("{")
                end_idx = answer.rfind("}") + 1
                json_str = answer[start_idx:end_idx]
                datos_ticket = json.loads(json_str)
                
                # Llamar al API de predicción en la EC2
                resultado_ml = predecir_costo_ticket(
                    evento=datos_ticket.get("evento"),
                    zona=datos_ticket.get("zona"),
                    cantidad=datos_ticket.get("cantidad")
                )
                
                if resultado_ml and "costo_estimado_pen" in resultado_ml:
                    costo = resultado_ml["costo_estimado_pen"]
                    evento_val = datos_ticket.get("evento")
                    zona_val = datos_ticket.get("zona")
                    cantidad_val = datos_ticket.get("cantidad")
                    metodo_val = datos_ticket.get("metodo_pago", "Yape")
                    dni_val = datos_ticket.get("documento_identidad", "12345678")

                    # Construimos un JSON limpio exclusivo para Swagger
                    swagger_payload = {
                        "evento": evento_val,
                        "zona": zona_val,
                        "cantidad": int(cantidad_val)
                    }
                    swagger_json_str = json.dumps(swagger_payload, indent=2, ensure_ascii=False)

                    # Conservamos el mensaje amigable, agregamos el costo total y anexamos el bloque JSON legible
                    answer = f"""¡Perfecto, ahora tenemos todos los datos!

📋 **Resumen de la reserva**
- **Evento:** {evento_val}
- **Zona:** {zona_val}
- **Cantidad:** {cantidad_val} entradas
- **Método de pago:** {metodo_val}
- **DNI:** {dni_val}

**Costo total estimado: S/ {costo}**

Todo está listo para proceder con la compra. Recibirás a continuación el comprobante y las instrucciones para completar el pago.

¡Gracias por confiar en TicketPass! 🎫 Si necesitas cualquier otra cosa, estoy aquí para ayudarte.

```json
{swagger_json_str}
```"""
        except Exception as e:
            print(f"Error procesando predicción o JSON: {str(e)}")
            pass

        timestamp = _now_iso()
        user_sk = f"MSG#{timestamp}#000#user"
        assistant_sk = f"MSG#{timestamp}#001#assistant"

        table.put_item(
            Item={
                "pk": f"CHAT#{chat_id}",
                "sk": user_sk,
                "role": "user",
                "content": prompt,
                "createdAt": timestamp,
            }
        )
        table.put_item(
            Item={
                "pk": f"CHAT#{chat_id}",
                "sk": assistant_sk,
                "role": "assistant",
                "content": answer,  # Guardado con el precio y el bloque JSON incluido
                "createdAt": timestamp,
            }
        )

        return _response(200, {"prompt": prompt, "answer": answer, "createdAt": timestamp})
    except Exception as exc:
        return _response(500, {"error": "send_message_failed", "detail": str(exc)})


def predecir_costo_ticket(evento: str, zona: str, cantidad: int) -> dict:
    url = "http://34.237.12.83:8000/predict"
    payload = {
        "evento": evento,
        "zona": zona,
        "cantidad": int(cantidad)
    }
    try:
        data_bytes = json.dumps(payload).encode("utf-8")
        req = Request(
            url, 
            data=data_bytes, 
            headers={"Content-Type": "application/json"}, 
            method="POST"
        )
        with urlopen(req, timeout=10) as response:
            if response.status == 200:
                return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Error conectando al API de ML: {str(e)}")
    return {}