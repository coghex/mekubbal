from __future__ import annotations

from typing import Any


def _ticker_recommendation_priority(label: str) -> int:
    return {
        "Bullish setup": 5,
        "Improving trend": 4,
        "Caution": 3,
        "Mixed trend": 2,
        "Avoid for now": 1,
    }.get(str(label).strip(), 0)


def _ticker_confidence_priority(label: str) -> int:
    return {"High": 3, "Medium": 2, "Low": 1}.get(str(label).strip(), 0)


def _ticker_status_priority(label: str) -> int:
    return {"Healthy": 2, "Watch": 1, "Critical": 0}.get(str(label).strip(), 0)


def _build_ticker_rankings(ticker_payload: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    scored_rows: list[dict[str, Any]] = []
    for symbol, payload in ticker_payload.items():
        active_vs_buy = payload.get("active_vs_buy_pct_value")
        active_vs_base = payload.get("active_vs_base_pct_value")
        momentum = payload.get("momentum")
        momentum_points = float(momentum) * 100.0 if momentum is not None else 0.0
        score = (
            _ticker_recommendation_priority(str(payload.get("recommendation", ""))) * 100.0
            + _ticker_confidence_priority(str(payload.get("confidence", ""))) * 20.0
            + _ticker_status_priority(str(payload.get("status", ""))) * 10.0
            + (float(active_vs_buy) if active_vs_buy is not None else 0.0)
            + ((float(active_vs_base) if active_vs_base is not None else 0.0) * 0.5)
            + (momentum_points * 0.25)
        )
        scored_rows.append(
            {
                "symbol": symbol,
                "symbol_category": payload.get("symbol_category"),
                "score": score,
                "recommendation": str(payload.get("recommendation", "")),
                "confidence": str(payload.get("confidence", "")),
                "status": str(payload.get("status", "")),
                "active_vs_buy_pct_text": str(payload.get("active_vs_buy_pct_text", "n/a")),
                "active_vs_buy_pct_value": active_vs_buy,
                "active_vs_base_pct_text": str(payload.get("active_vs_base_pct_text", "n/a")),
                "active_vs_base_pct_value": active_vs_base,
                "momentum_points": momentum_points if momentum is not None else None,
            }
        )

    scored_rows.sort(
        key=lambda row: (
            -float(row["score"]),
            -(float(row["active_vs_buy_pct_value"]) if row["active_vs_buy_pct_value"] is not None else -10**9),
            -(float(row["active_vs_base_pct_value"]) if row["active_vs_base_pct_value"] is not None else -10**9),
            str(row["symbol"]),
        )
    )

    rankings: list[dict[str, Any]] = []
    for index, row in enumerate(scored_rows):
        symbol = str(row["symbol"])
        active_vs_buy_text = str(row["active_vs_buy_pct_text"])
        active_vs_base_text = str(row["active_vs_base_pct_text"])
        recommendation = str(row["recommendation"])
        confidence = str(row["confidence"])
        status = str(row["status"])
        momentum_points = row.get("momentum_points")
        reasons = [
            f"{symbol} ranks #{index + 1} because it is {recommendation.lower()} with {confidence.lower()} confidence.",
            f"It is running at {active_vs_buy_text} versus buy-and-hold and {active_vs_base_text} versus the base setup.",
        ]
        if momentum_points is not None:
            if float(momentum_points) > 0.2:
                reasons.append("Its recent daily edge is improving.")
            elif float(momentum_points) < -0.2:
                reasons.append("Its recent daily edge has cooled lately.")
        if status == "Critical":
            reasons.append("Current warning flags are dragging it down the list.")
        elif status == "Watch":
            reasons.append("It stays in the middle of the list because some caution flags are still active.")
        else:
            reasons.append("It stays near the top because there are no active drift warnings.")
        reason = " ".join(reasons)

        if index == 0 and len(scored_rows) > 1:
            next_row = scored_rows[index + 1]
            comparison = (
                f"Ahead of {next_row['symbol']} because its overall signal is stronger: "
                f"{active_vs_buy_text} versus buy-and-hold compared with {next_row['active_vs_buy_pct_text']}, "
                f"and {confidence.lower()} confidence versus {str(next_row['confidence']).lower()} confidence."
            )
        elif index > 0:
            prev_row = scored_rows[index - 1]
            comparison = (
                f"Trailing {prev_row['symbol']} because its current edge is weaker: "
                f"{active_vs_buy_text} versus buy-and-hold compared with {prev_row['active_vs_buy_pct_text']}, "
                f"with {confidence.lower()} confidence versus {str(prev_row['confidence']).lower()} confidence."
            )
        else:
            comparison = "This ticker currently leads the list."

        ranking_row = {
            "rank": index + 1,
            "symbol": symbol,
            "symbol_category": row.get("symbol_category"),
            "ranking_score": float(row["score"]),
            "reason": reason,
            "comparison": comparison,
        }
        rankings.append(ranking_row)
        ticker_payload[symbol]["priority_rank"] = index + 1
        ticker_payload[symbol]["priority_reason"] = reason
        ticker_payload[symbol]["comparison_summary"] = comparison
        ticker_payload[symbol]["ranking_score"] = float(row["score"])
    return rankings
