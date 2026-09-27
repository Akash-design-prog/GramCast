"""Alert-sending script (docs/ISSUES_PLAN.md item 4b): checks today's (the dataset's latest real
date's) prediction for every registered farmer against real IMD-aligned thresholds, and sends a
WhatsApp template message to anyone whose forecast reaches MODERATE or above - not every farmer every
day, to avoid alert fatigue on days with little/no rain.

Requires a pre-approved WhatsApp message template (see whatsapp_client.send_template_message's own
docstring: Meta requires templates be created and approved in the dashboard before this can message a
farmer outside the 24h customer-service window - this script cannot create that template, only send an
already-approved one). Meant to run manually or via a scheduled job (cron/Task Scheduler); no FastAPI/
server dependency, same as the data_pipeline scripts' own plain run()+main() pattern.

Run: .venv/Scripts/python.exe backend/whatsapp_alerts.py [--date YYYY-MM-DD] [--dry-run]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from advisory.rules import RainCategory, classify_rainfall  # noqa: E402
from inference import GramCastInference  # noqa: E402
from villages import VillageIndex, panchayat_value_by_index  # noqa: E402
import whatsapp_client  # noqa: E402
import whatsapp_registry  # noqa: E402

ALERT_TEMPLATE_NAME = "gramcast_rain_alert"  # must already exist + be approved in the Meta dashboard
ALERT_TEMPLATE_LANGUAGE = "en"
ALERT_MIN_CATEGORY = RainCategory.MODERATE  # only alert for MODERATE and above - skip no_rain/light

_CATEGORY_ORDER = [RainCategory.NO_RAIN, RainCategory.LIGHT, RainCategory.MODERATE, RainCategory.HEAVY]


def send_alerts(inference: GramCastInference, village_index: VillageIndex, date: str, dry_run: bool = False) -> list[dict]:
    """Returns every alert that was sent (or would be, if dry_run) - useful for a demo without spending
    real WhatsApp API calls, or before real credentials even exist yet."""
    pred = inference.predict_day(date)

    sent = []
    for farmer in whatsapp_registry.list_farmers():
        try:
            row_index = village_index.find_row_index_for(farmer["village_name"], farmer["sub_district"])
            p50_mm = panchayat_value_by_index(pred["p50"], row_index)["mean_mm"]
        except ValueError:
            continue  # this farmer's registered village has no coverage for this date - skip, don't crash the whole run

        category = classify_rainfall(p50_mm)
        if _CATEGORY_ORDER.index(category) < _CATEGORY_ORDER.index(ALERT_MIN_CATEGORY):
            continue

        record = {"phone": farmer["phone"], "village_name": farmer["village_name"], "category": category.value, "p50_mm": round(p50_mm, 1)}
        if not dry_run:
            whatsapp_client.send_template_message(
                to=farmer["phone"],
                template_name=ALERT_TEMPLATE_NAME,
                language_code=ALERT_TEMPLATE_LANGUAGE,
                body_params=[farmer["village_name"], f"{p50_mm:.1f}", category.value.replace("_", " ")],
            )
        sent.append(record)
    return sent


def main() -> None:
    parser = argparse.ArgumentParser(description="Send WhatsApp rain alerts to registered farmers.")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD, defaults to the dataset's latest available date")
    parser.add_argument("--dry-run", action="store_true", help="Compute and print alerts without actually sending them")
    args = parser.parse_args()

    if not args.dry_run and not whatsapp_client.is_configured():
        print("WHATSAPP_ACCESS_TOKEN/WHATSAPP_PHONE_NUMBER_ID not set - run with --dry-run, or set them first.")
        return

    inference = GramCastInference()
    village_index = VillageIndex()
    date = args.date or inference.sorted_available_dates()[-1]

    sent = send_alerts(inference, village_index, date, dry_run=args.dry_run)
    print(f"{'Would send' if args.dry_run else 'Sent'} {len(sent)} alert(s) for {date}:")
    for r in sent:
        print(f"  {r['village_name']} -> {r['phone']}: {r['category']} ({r['p50_mm']}mm)")


if __name__ == "__main__":
    main()
