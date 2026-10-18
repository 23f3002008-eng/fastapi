
import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai

app = FastAPI(title="Acme Appliances Order Ledger")

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)


RATES = {"USD": 1.0, "EUR": 1.1, "INR": 0.01126}
TZ = ZoneInfo("Asia/Kolkata")


def parse_time(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt.astimezone(TZ)


def load_orders():
    latest = {}

    with open("export_download.bin", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            order = json.loads(line)
            oid = order["id"]

            if (
                oid not in latest
                or parse_time(order["updated_at"])
                > parse_time(latest[oid]["updated_at"])
            ):
                latest[oid] = order

    return list(latest.values())


ORDERS = load_orders()


class Question(BaseModel):
    question: str



def interpret_question(question):
    prompt = """
You are a query interpreter for the Acme Appliances order ledger.

Convert the user's question into a JSON object with these fields:
- metric: revenue, count, or list
- product: exact product name or null
- region: exact region or null
- status: paid, refunded, void, or null
- currency: USD, EUR, INR, or null
- start_date: YYYY-MM-DD or null
- end_date: YYYY-MM-DD or null
- date_field: created_at or updated_at
- customer: customer ID such as C1018, or null
- group_by: product, region, currency, status, or null

Rules:
- Revenue and sales questions use metric "revenue".
- "How many orders" questions use metric "count".
- Listing questions use metric "list".
- Interpret dates in the Asia/Kolkata timezone.
- Only include filters supported by the question.
- Use null for unspecified fields.
- Return JSON only, without Markdown.

User question:
""" + question

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "temperature": 0,
        },
    )

    return json.loads(response.text)



def answer_question(question):
    spec = interpret_question(question)
    rows = ORDERS.copy()
    

    if spec.get("product"):
        requested = spec["product"].strip().lower()

        # Match plural and singular forms against actual ledger names.
        products = sorted(
            {o["product"] for o in ORDERS},
            key=len
        )

        canonical = next(
            (
                product for product in products
                if product.lower() == requested
                or product.lower() + "s" == requested
                or product.lower() + "es" == requested
            ),
            None
        )

        if canonical is None:
            canonical = next(
                (
                    product for product in products
                    if requested in product.lower()
                    or product.lower() in requested
                ),
                None
            )

        if canonical is None:
            raise ValueError(
                f"Unknown product: {spec['product']}"
            )

        rows = [
            o for o in rows
            if o["product"].lower() == canonical.lower()
        ]


    if spec.get("region"):
        rows = [
            o for o in rows
            if o["region"].lower() == spec["region"].lower()
        ]

    if spec.get("status"):
        rows = [
            o for o in rows
            if o["status"].lower() == spec["status"].lower()
        ]

    if spec.get("customer"):
        rows = [
            o for o in rows
            if o["customer"].lower() == spec["customer"].lower()
        ]

    if spec.get("start_date") or spec.get("end_date"):
        field = spec.get("date_field", "created_at")

        if field not in ("created_at", "updated_at"):
            field = "created_at"

        if spec.get("start_date"):
            start = datetime.fromisoformat(
                spec["start_date"]
            ).replace(tzinfo=TZ)

            rows = [
                o for o in rows
                if parse_time(o[field]) >= start
            ]

        if spec.get("end_date"):
            end = datetime.fromisoformat(
                spec["end_date"]
            ).replace(tzinfo=TZ)

            # Include the entire end date.
            from datetime import timedelta
            end += timedelta(days=1)

            rows = [
                o for o in rows
                if parse_time(o[field]) < end
            ]

    metric = spec.get("metric", "count")

    if metric == "count":
        return len(rows)

    if metric == "list":
        return rows[:100]

    # Revenue counts paid orders only, regardless of the
    # status filter requested in the question.
    paid = [o for o in rows if o["status"] == "paid"]

    currency = spec.get("currency")

    if not currency:
        currency = "USD"

    currency = str(currency).strip().upper()

    if currency not in RATES:
        currency = "USD"

    total = sum(
        float(o["amount"]) * RATES[o["currency"]]
        for o in paid
    )

    return round(total / RATES[currency], 2)


@app.post("/answer")
def answer(request: Question):
    try:
        result = answer_question(request.question)
        return {"answer": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
