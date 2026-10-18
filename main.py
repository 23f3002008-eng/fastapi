
import os
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai

app = FastAPI(title="Acme Appliances Order Ledger")

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# USD value of one unit of each currency
RATES = {
    "USD": 1.0,
    "EUR": 1.1,
    "INR": 0.01126,
}

TZ = ZoneInfo("Asia/Kolkata")

# Resolve the data file relative to main.py, not the working directory.
DATA_FILE = Path(__file__).resolve().parent / "export_download.bin"


def parse_time(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))

    # Treat timestamps without timezone information as UTC.
    if dt.tzinfo is None:
        from datetime import timezone
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(TZ)


def load_orders():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}. "
            "Ensure export_download.bin is included in the deployment."
        )

    latest = {}

    with DATA_FILE.open("r", encoding="utf-8") as f:
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
You interpret questions about the Acme Appliances order ledger.
Return ONLY a valid JSON object with these fields:

{
  "metric": "revenue" | "count" | "list" | "average_order_value",
  "product": string or null,
  "region": string or null,
  "status": "paid" | "refunded" | "void" or null,
  "currency": "USD" | "EUR" | "INR" or null,
  "start_date": "YYYY-MM-DD" or null,
  "end_date": "YYYY-MM-DD" or null,
  "date_field": "created_at" | "updated_at",
  "customer": string or null,
  "group_by": "product" | "region" | "currency" | "status" or null
}

Rules:
- "total revenue", "total sales", and "sales amount" mean metric="revenue".
- "average order value", "average paid order value", "mean order value",
  and "average spend per order" mean metric="average_order_value".
- "how many orders" means metric="count".
- Requests to list orders mean metric="list".
- If the question says "paid orders", set status="paid".
- Extract explicitly mentioned product, region, customer, and date filters.
- Do not invent filters that are not in the question.
- Use the requested output currency when specified.
- Interpret dates in the Asia/Kolkata timezone.
- Use date_field="created_at" unless the question explicitly refers to
  updates or modification dates.
- Return null for unspecified optional filters.
- Do not answer the question; only interpret it.

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

    spec = json.loads(response.text)

    # Deterministic overrides for important phrases. This prevents
    # "average order value" from being treated as total revenue.
    q = question.lower()

    if re.search(
        r"\b(average|avg|mean)\b.*\b(order value|order amount|order|orders)\b"
        r"|\b(order value|order amount)\b.*\b(average|avg|mean)\b",
        q,
    ):
        spec["metric"] = "average_order_value"

    if re.search(r"\bhow many\b|\bcount of\b|\bnumber of orders\b", q):
        spec["metric"] = "count"

    if re.search(r"\b(list|show me the orders|display the orders)\b", q):
        spec["metric"] = "list"

    # Explicit region mentioned by the user takes precedence.
    regions = {
        str(order["region"]).strip().lower(): order["region"]
        for order in ORDERS
        if order.get("region") is not None
    }

    for region_lower, region_original in regions.items():
        if re.search(r"\b" + re.escape(region_lower) + r"\b", q):
            spec["region"] = region_original
            break

    # Explicit status phrases take precedence.
    for status in ("paid", "refunded", "void"):
        if re.search(r"\b" + status + r"\b", q):
            spec["status"] = status
            break

    # An average of paid order values must use paid orders only.
    if spec["metric"] == "average_order_value" and "paid" in q:
        spec["status"] = "paid"

    # Infer currency from the question if explicitly mentioned.
    for currency in RATES:
        if re.search(r"\b" + currency.lower() + r"\b", q):
            spec["currency"] = currency
            break

    return spec


def answer_question(question):
    spec = interpret_question(question)
    rows = ORDERS.copy()

    # Product filter
    if spec.get("product"):
        requested = str(spec["product"]).strip().lower()
        products = sorted(
            {o["product"] for o in ORDERS},
            key=len,
        )

        canonical = next(
            (
                product for product in products
                if product.lower() == requested
            ),
            None,
        )

        if canonical is None:
            canonical = next(
                (
                    product for product in products
                    if requested in product.lower()
                    or product.lower() in requested
                ),
                None,
            )

        if canonical is None:
            raise ValueError(f"Unknown product: {spec['product']}")

        rows = [
            o for o in rows
            if o["product"].lower() == canonical.lower()
        ]

    # Region filter
    if spec.get("region"):
        region = str(spec["region"]).strip().lower()
        rows = [
            o for o in rows
            if str(o["region"]).strip().lower() == region
        ]

    # Status filter
    if spec.get("status"):
        status = str(spec["status"]).strip().lower()
        rows = [
            o for o in rows
            if str(o["status"]).strip().lower() == status
        ]

    # Customer filter
    if spec.get("customer"):
        customer = str(spec["customer"]).strip().lower()
        rows = [
            o for o in rows
            if str(o["customer"]).strip().lower() == customer
        ]

    # Date filters: inclusive start and inclusive end date.
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
            ).replace(tzinfo=TZ) + timedelta(days=1)

            rows = [
                o for o in rows
                if parse_time(o[field]) < end
            ]

    metric = spec.get("metric", "count")

    if metric == "count":
        return len(rows)

    if metric == "list":
        return rows[:100]

    currency = str(spec.get("currency") or "USD").strip().upper()

    if currency not in RATES:
        raise ValueError(f"Unsupported currency: {currency}")

    # Revenue and average order value use paid orders only.
    # Convert each order into USD before aggregation.
    paid_orders = [
        o for o in rows
        if str(o["status"]).strip().lower() == "paid"
    ]

    total_usd = sum(
        float(o["amount"]) * RATES[str(o["currency"]).upper()]
        for o in paid_orders
    )

    if metric == "average_order_value":
        if not paid_orders:
            return 0.0

        # Average = total converted paid revenue / paid order count.
        result_in_usd = total_usd / len(paid_orders)
    elif metric == "revenue":
        result_in_usd = total_usd
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Convert from USD into the requested output currency.
    result = result_in_usd / RATES[currency]

    return round(result, 2)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded_orders": len(ORDERS),
        "dataset_exists": DATA_FILE.exists(),
    }


@app.post("/answer")
def answer(request: Question):
    try:
        result = answer_question(request.question)
        return {"answer": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
