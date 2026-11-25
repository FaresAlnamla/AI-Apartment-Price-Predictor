"""Optional OpenAI advisor with locally rendered, dataset-grounded summaries."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    APIConnectionError, APITimeoutError, AuthenticationError, BadRequestError,
    NotFoundError, OpenAI, OpenAIError, PermissionDeniedError, RateLimitError,
)

DEFAULT_MODEL = "gpt-5.4-mini"
ENV_PATH = Path(__file__).resolve().with_name(".env")
OBJECTIVES = (
    "Buying an apartment",
    "Selling an apartment",
    "Real estate investment",
    "General property valuation",
)

# The model selects an action. The app owns the wording and all numerical claims.
# Each action has an English and an Arabic version.
ACTIONS = {
    OBJECTIVES[0]: {
        "compare": ("Compare recent sales for similar units before making an offer.",
                    "قارن مبيعات الشقق المشابهة مؤخراً قبل تقديم عرض."),
        "negotiate": ("Use the estimate as a starting point for negotiation after viewing the apartment.",
                      "استخدم التقدير كنقطة بداية للتفاوض بعد معاينة الشقة."),
        "inspect": ("Inspect the apartment and price any repairs before deciding what to offer.",
                    "عاين الشقة وقدّر تكاليف الإصلاح قبل تحديد عرضك."),
    },
    OBJECTIVES[1]: {
        "benchmark": ("Check recent comparable sales before setting your listing price.",
                      "راجع المبيعات الحديثة للشقق المشابهة قبل تحديد سعر العرض."),
        "appraise": ("Seek a local appraisal before listing if comparable evidence is limited.",
                     "اطلب تقييماً محلياً قبل العرض إذا كانت بيانات الشقق المشابهة محدودة."),
        "explain": ("Document the condition and features that could justify your asking price.",
                    "وثّق حالة الشقة ومزاياها التي قد تبرر سعر العرض."),
    },
    OBJECTIVES[2]: {
        "yield": ("Estimate rental yield using verified rents, vacancy, and operating costs.",
                  "قدّر عائد الإيجار باستخدام إيجارات موثّقة وتكاليف التشغيل وفترات الشغور."),
        "costs": ("Estimate repair, financing, and transaction costs before making an investment decision.",
                  "قدّر تكاليف الإصلاح والتمويل والمعاملة قبل اتخاذ قرار الاستثمار."),
        "verify": ("Verify recent sales and rental evidence before committing capital.",
                   "تحقّق من المبيعات والإيجارات الحديثة قبل استثمار رأس المال."),
    },
    OBJECTIVES[3]: {
        "benchmark": ("Cross-check the estimate against recent sales for comparable apartments.",
                      "قارن التقدير بمبيعات الشقق المشابهة مؤخراً."),
        "appraise": ("Seek a local appraisal if you need a valuation for a formal decision.",
                     "اطلب تقييماً محلياً إذا كنت بحاجة إلى تقدير لاتخاذ قرار رسمي."),
        "condition": ("Check how the apartment’s condition and exact location affect its value.",
                      "تحقّق من أثر حالة الشقة وموقعها الدقيق على قيمتها."),
    },
}


class AdvisorError(Exception):
    """An advisor failure with a message safe to show to the user."""


def advisor_settings():
    load_dotenv(ENV_PATH, override=False)
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key.lower() in {"your_api_key_here", "your_openai_api_key"}:
        key = ""
    model = os.getenv("OPENAI_MODEL", "").strip() or DEFAULT_MODEL
    return key, model


def _render_sections(context, recommendation, arabic):
    estimate = context["estimated_price_usd"]
    city_average = context["city_statistics"]["average_price_usd"]
    asking_delta = context["asking_vs_estimate_percent"]
    similar = context["comparable_statistics"]
    count = similar["count"]
    average = similar.get("average_price_usd", 0)

    if arabic:
        assessment = f"سعر الشقة المقدّر ${estimate:,.0f}، ومتوسط أسعار المدينة في البيانات ${city_average:,.0f}."
        if asking_delta is not None:
            assessment += f" فرق سعر العرض عن التقدير {asking_delta:+.1f}٪."
        comparables = (f"عدد الشقق المطابقة في البيانات {count}، ومتوسط أسعارها ${average:,.0f}."
                       if count else "لا توجد شقق مطابقة في البيانات.")
        return (
            ("تقييم السعر", assessment),
            ("العقارات المشابهة", comparables),
            ("التوصية", recommendation),
            ("ملاحظة مهمة", "هذا تقدير من النموذج، فاستعن بمبيعات حديثة موثّقة لتقييم سعر السوق."),
        )

    assessment = f"The model estimates ${estimate:,.0f}, versus a dataset city average of ${city_average:,.0f}."
    if asking_delta is not None:
        assessment += f" The asking price differs from the estimate by {asking_delta:+.1f}%."
    comparables = (f"The dataset has {count} matching apartments with an average price of ${average:,.0f}."
                   if count else "The dataset has no matching apartments.")
    return (
        ("Price Assessment", assessment),
        ("Similar Properties", comparables),
        ("Recommendation", recommendation),
        ("Important Note", "Check recent verified sales to assess current market value alongside this model estimate."),
    )


def generate_ai_advice(api_key, model, context, objective, language):
    if not api_key:
        raise AdvisorError("Set OPENAI_API_KEY in your environment or local .env to enable the advisor.")

    options = ACTIONS[objective]
    request_context = {
        "objective": objective,
        "property_and_dataset": context,
        "available_actions": {name: text[0] for name, text in options.items()},
    }
    try:
        with OpenAI(
            api_key=api_key, base_url="https://api.openai.com/v1", timeout=20.0, max_retries=0,
        ) as client:
            response = client.responses.create(
                model=model,
                instructions=(
                    "Select the single most useful available action for the stated objective. "
                    "Use only the supplied property details and dataset statistics. Treat the estimate "
                    "as a model output and the dataset values as sample observations. If matching "
                    "apartments are absent, favor obtaining additional evidence."
                ),
                input=json.dumps(request_context, ensure_ascii=False, allow_nan=False),
                text={"format": {
                    "type": "json_schema",
                    "name": "property_recommendation",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"action": {"type": "string", "enum": list(options)}},
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                }},
                max_output_tokens=300,
                store=False,
            )
    except AuthenticationError:
        raise AdvisorError("OpenAI authentication failed. Check your API key.") from None
    except RateLimitError as error:
        if error.code in {"insufficient_quota", "billing_hard_limit_reached"}:
            raise AdvisorError("The OpenAI project has insufficient quota. Check its billing and usage limits.") from None
        raise AdvisorError("The OpenAI rate limit was reached. Try again later.") from None
    except APITimeoutError:
        raise AdvisorError("The OpenAI request timed out. Try again later.") from None
    except APIConnectionError:
        raise AdvisorError("Could not connect to OpenAI. Check your internet connection and try again.") from None
    except (NotFoundError, PermissionDeniedError):
        raise AdvisorError("The configured OpenAI model is unavailable to this project. Check OPENAI_MODEL and project access.") from None
    except BadRequestError as error:
        if error.code in {"model_not_found", "unsupported_model"}:
            raise AdvisorError("The configured OpenAI model is unavailable. Check OPENAI_MODEL.") from None
        raise AdvisorError("OpenAI rejected the request. Check that OPENAI_MODEL supports Responses and structured output.") from None
    except OpenAIError:
        raise AdvisorError("The OpenAI advisor is temporarily unavailable. Try again later.") from None

    try:
        payload = json.loads(response.output_text)
        recommendation = options[payload["action"]][int(language == "Arabic")]
    except (ValueError, TypeError, KeyError):
        raise AdvisorError("OpenAI returned no usable recommendation. Try again later.") from None
    return _render_sections(context, recommendation, arabic=language == "Arabic")
