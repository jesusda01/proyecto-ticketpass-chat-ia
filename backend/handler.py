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

        # Interceptar si el bot devolvió un bloque JSON para cotización o modelos de ML
        try:
            if "{" in answer and "}" in answer:
                start_idx = answer.find("{")
                end_idx = answer.rfind("}") + 1
                json_str = answer[start_idx:end_idx]
                datos_modelo = json.loads(json_str)
                
                # CASO 1: Cotización tradicional (Puerto 8000)
                if "evento" in datos_modelo and "zona" in datos_modelo and "cantidad" in datos_modelo:
                    resultado_ml = predecir_costo_ticket(
                        evento=datos_modelo.get("evento"),
                        zona=datos_modelo.get("zona"),
                        cantidad=datos_modelo.get("cantidad")
                    )
                    
                    if resultado_ml and "costo_estimado_pen" in resultado_ml:
                        costo = resultado_ml["costo_estimado_pen"]
                        evento_val = datos_modelo.get("evento")
                        zona_val = datos_modelo.get("zona")
                        cantidad_val = datos_modelo.get("cantidad")
                        metodo_val = datos_modelo.get("metodo_pago", "Yape")
                        dni_val = datos_modelo.get("documento_identidad", "12345678")

                        swagger_payload = {
                            "evento": evento_val,
                            "zona": zona_val,
                            "cantidad": int(cantidad_val)
                        }
                        swagger_json_str = json.dumps(swagger_payload, indent=2, ensure_ascii=False)

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

                # CASO 2: Regresión de Ventas (Puerto 8001 - /ventas)
                elif "artista" in datos_modelo and "aforo_maximo" in datos_modelo:
                    resultado_ventas = predecir_ventas_concierto(datos_modelo)
                    if resultado_ventas and "entradas_estimadas_vendidas" in resultado_ventas:
                        entradas = resultado_ventas["entradas_estimadas_vendidas"]
                        moneda = resultado_ventas.get("moneda", "PEN")
                        
                        answer = f"""📊 **Predicción de Ventas Estimadas (Machine Learning)**

- **Artista / Evento:** {datos_modelo.get('artista')}
- **Recinto:** {datos_modelo.get('recinto', 'N/D')}
- **Entradas estimadas a vender:** **{entradas} unidades** ({moneda})

El modelo de regresión ha procesado las variables de aforo, oyentes y antelación para calcular esta proyección. 🚀
"""

                # CASO 3: Clasificación Sold Out (Puerto 8001 - /sold_out)
                elif "artista" in datos_modelo and "oyentes_spotify_peru" in datos_modelo and "presupuesto_marketing_digital_usd" in datos_modelo:
                    resultado_soldout = predecir_sold_out_concierto(datos_modelo)
                    if resultado_soldout:
                        es_sold_out = resultado_soldout.get("es_sold_out", False)
                        probabilidad = resultado_soldout.get("probabilidad_exito", 0)
                        estado_texto = "¡Sí! Habrá Sold Out 🔥" if es_sold_out else "No se proyecta Sold Out 📉"
                        
                        answer = f"""🔮 **Evaluación de Probabilidad de Sold Out (Machine Learning)**

- **Artista:** {datos_modelo.get('artista')}
- **¿Habrá Sold Out?:** **{estado_texto}**
- **Probabilidad de éxito:** **{probabilidad}%**

El modelo de clasificación ha analizado el impacto en marketing y streaming para este lanzamiento.
"""

        except Exception as e:
            print(f"Error procesando predicción/cotización o JSON: {str(e)}")
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
                "content": answer,
                "createdAt": timestamp,
            }
        )

        return _response(200, {"prompt": prompt, "answer": answer, "createdAt": timestamp})
    except Exception as exc:
        return _response(500, {"error": "send_message_failed", "detail": str(exc)})


def predecir_costo_ticket(evento: str, zona: str, cantidad: int) -> dict:
    url = "http://34.237.12.83:8000/quotation"
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
        print(f"Error conectando al API de Cotización (8000): {str(e)}")
    return {}


def predecir_ventas_concierto(payload: dict) -> dict:
    url = "http://34.237.12.83:8001/ventas"
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
        print(f"Error conectando al API de Ventas (8001): {str(e)}")
    return {}


def predecir_sold_out_concierto(payload: dict) -> dict:
    url = "http://34.237.12.83:8001/sold_out"
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
        print(f"Error conectando al API de Sold Out (8001): {str(e)}")
    return {}