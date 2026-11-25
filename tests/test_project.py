import json
import math
import os
from pathlib import Path
import tomllib
import unittest
from unittest.mock import patch

import httpx
import numpy as np
import pandas as pd
from openai import (
    APIConnectionError, APITimeoutError, AuthenticationError, BadRequestError,
    InternalServerError, NotFoundError, PermissionDeniedError, RateLimitError,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

from advisor import AdvisorError, DEFAULT_MODEL, OBJECTIVES, advisor_settings, generate_ai_advice
from valuation import (
    FEATURES, find_similar_apartments, get_statistics, load_dataset, predict_price, train_model,
)

ROOT = Path(__file__).resolve().parents[1]
TEST_KEY = "mock-test-key"


class ValuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset()
        cls.model, cls.metrics = train_model(cls.data)

    def test_dataset_and_metrics(self):
        self.assertTrue(self.data["apartment_id"].is_unique)
        self.assertGreater(self.data["city"].nunique(), 1)
        self.assertEqual(self.metrics["test_rows"], math.ceil(len(self.data) * 0.20))
        self.assertEqual(self.metrics["train_rows"] + self.metrics["test_rows"], len(self.data))
        self.assertGreater(self.metrics["r2"], 0)
        self.assertGreater(self.metrics["mae"], 0)
        self.assertGreaterEqual(self.metrics["rmse"], self.metrics["mae"])

    def test_invalid_dataset_rejected(self):
        with patch("valuation.pd.read_csv", return_value=self.data.drop(columns=["city"])):
            with self.assertRaisesRegex(ValueError, "Missing dataset columns"):
                load_dataset()
        invalid = self.data.assign(area_sqm=0)
        with patch("valuation.pd.read_csv", return_value=invalid):
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                load_dataset()

    def test_encoding_and_original_model_parity(self):
        encoded = pd.get_dummies(self.data[[*FEATURES, "price_usd"]], columns=["city"], dtype=int)
        original_x = encoded.drop(columns="price_usd")
        x_train, _, y_train, _ = train_test_split(
            original_x, encoded["price_usd"], test_size=0.20, random_state=42
        )
        original = RandomForestRegressor(random_state=42).fit(x_train, y_train)
        np.testing.assert_allclose(
            self.model.predict(self.data[FEATURES]),
            original.predict(original_x),
            rtol=0, atol=0,
        )
        for city in self.data["city"].unique():
            self.assertGreater(predict_price(self.model, city, 140, 3, 3), 0)

    def test_comparables_and_statistics(self):
        data = pd.DataFrame({
            "city": ["A", "A", "A", "A", "A", "B"],
            "area_sqm": [120, 100, 80, 121, 100, 100],
            "rooms": [3, 3, 3, 3, 4, 3],
            "price_usd": [120000, 95000, 72000, 121000, 100000, 99000],
        })
        similar = find_similar_apartments(data, "A", 100, 3)
        self.assertEqual(list(similar["area_sqm"]), [100, 120, 80])
        self.assertEqual(list(similar["price_per_sqm"]), [950, 1000, 900])
        self.assertEqual(get_statistics(similar), {
            "count": 3, "average_price_usd": 95666.67, "median_price_usd": 95000,
            "minimum_price_usd": 72000, "maximum_price_usd": 120000,
            "average_area_sqm": 100, "average_price_per_sqm": 950,
        })
        empty = find_similar_apartments(data, "A", 100, 6)
        self.assertTrue(empty.empty)
        self.assertEqual(get_statistics(empty), {"count": 0})


class AdvisorTests(unittest.TestCase):
    def setUp(self):
        self.context = {
            "estimated_price_usd": 120000,
            "asking_vs_estimate_percent": 4.25,
            "city_statistics": {"average_price_usd": 125000},
            "comparable_statistics": {"count": 2, "average_price_usd": 122000},
        }
        client = patch("advisor.OpenAI")
        self.client = client.start()
        self.addCleanup(client.stop)
        self.create = self.client.return_value.__enter__.return_value.responses.create
        self.create.return_value.output_text = '{"action":"compare"}'

    def request(self, language="English", key=TEST_KEY):
        return generate_ai_advice(key, DEFAULT_MODEL, self.context, OBJECTIVES[0], language)

    def test_env_missing_placeholder_and_precedence(self):
        with patch("advisor.load_dotenv") as load, patch.dict(os.environ, {}, clear=True):
            self.assertEqual(advisor_settings(), ("", DEFAULT_MODEL))
            with patch.dict(os.environ, {"OPENAI_API_KEY": "your_api_key_here"}):
                self.assertEqual(advisor_settings(), ("", DEFAULT_MODEL))
            with patch.dict(os.environ, {"OPENAI_API_KEY": TEST_KEY, "OPENAI_MODEL": "custom-model"}):
                self.assertEqual(advisor_settings(), (TEST_KEY, "custom-model"))
            self.assertFalse(load.call_args.kwargs["override"])

    def test_missing_key_never_creates_client(self):
        with self.assertRaisesRegex(AdvisorError, "Set OPENAI_API_KEY"):
            self.request(key="")
        self.client.assert_not_called()

    def test_bilingual_grounded_sections_and_request(self):
        headings = {
            "English": ["Price Assessment", "Similar Properties", "Recommendation", "Important Note"],
            "Arabic": ["تقييم السعر", "العقارات المشابهة", "التوصية", "ملاحظة مهمة"],
        }
        for language, expected in headings.items():
            with self.subTest(language=language):
                sections = self.request(language)
                self.assertEqual([title for title, _ in sections], expected)
                self.assertIn("120,000", sections[0][1])
                self.assertIn("122,000", sections[1][1])
                self.assertLessEqual(sum(len(f"{title} {body}".split()) for title, body in sections), 120)
                for _, body in sections:
                    self.assertLessEqual(len(body.rstrip(".").split(". ")), 2)
        call = self.create.call_args.kwargs
        self.assertEqual(call["model"], DEFAULT_MODEL)
        self.assertFalse(call["store"])
        self.assertTrue(call["text"]["format"]["strict"])
        self.assertEqual(json.loads(call["input"])["property_and_dataset"], self.context)
        self.assertNotIn(TEST_KEY, call["input"])

    def test_no_comparables_and_invalid_response(self):
        self.context["comparable_statistics"] = {"count": 0}
        self.context["asking_vs_estimate_percent"] = None
        self.assertIn("no matching apartments", self.request()[1][1])
        for output in ("", "[]", '{"action":"invented"}', '{"action":[]}'):
            with self.subTest(output=output):
                self.create.return_value.output_text = output
                with self.assertRaisesRegex(AdvisorError, "no usable recommendation"):
                    self.request()

    def test_api_failures_are_safe_and_distinct(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")

        def status_error(error_type, status, code=None):
            return error_type("sensitive test detail", response=httpx.Response(status, request=request),
                              body={"code": code})

        cases = (
            (status_error(AuthenticationError, 401), "authentication"),
            (status_error(NotFoundError, 404), "model is unavailable"),
            (status_error(PermissionDeniedError, 403), "model is unavailable"),
            (status_error(BadRequestError, 400, "unsupported_model"), "model is unavailable"),
            (status_error(BadRequestError, 400), "rejected"),
            (status_error(RateLimitError, 429, "insufficient_quota"), "insufficient quota"),
            (status_error(RateLimitError, 429), "rate limit"),
            (APITimeoutError(request=request), "timed out"),
            (APIConnectionError(request=request), "connect"),
            (status_error(InternalServerError, 500), "temporarily unavailable"),
        )
        for error, message in cases:
            with self.subTest(message=message):
                self.create.side_effect = error
                with self.assertRaisesRegex(AdvisorError, message) as raised:
                    self.request()
                self.assertNotIn("sensitive test detail", str(raised.exception))


class StreamlitTests(unittest.TestCase):
    def app(self):
        result = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
        self.assertFalse(result.exception)
        return result

    def test_light_theme_and_default_prediction(self):
        theme = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())["theme"]
        self.assertEqual(theme["base"], "light")
        self.assertEqual(theme["primaryColor"], "#176b70")
        app = self.app()
        self.assertFalse(app.checkbox[0].value)
        app.number_input[3].set_value(135000)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("Estimated Apartment Price", [metric.label for metric in app.metric])
        self.assertIn("Asking Price", [metric.label for metric in app.metric])
        self.assertEqual(len(app.get("vega_lite_chart")), 1)
        self.assertEqual(len(app.dataframe), 1)
        self.assertIn("R²:", app.expander[0].markdown[0].value)

    def test_no_matches_and_missing_key_leave_valuation_available(self):
        app = self.app()
        app.number_input[0].set_value(75)
        app.number_input[1].set_value(6)
        app.checkbox[0].check()
        with patch("advisor.load_dotenv"), patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("Estimated Apartment Price", [metric.label for metric in app.metric])
        self.assertTrue(any("No closely matching" in warning.value for warning in app.warning))
        self.assertTrue(any("Set OPENAI_API_KEY" in warning.value for warning in app.warning))

    def test_advisor_failure_leaves_valuation_available(self):
        app = self.app()
        app.checkbox[0].check()
        with patch("advisor.OpenAI") as client, patch.dict(os.environ, {"OPENAI_API_KEY": TEST_KEY}):
            create = client.return_value.__enter__.return_value.responses.create
            create.side_effect = APITimeoutError(request=httpx.Request("POST", "https://api.openai.com"))
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("Estimated Apartment Price", [metric.label for metric in app.metric])
        self.assertTrue(any("timed out" in warning.value for warning in app.warning))

    def test_arabic_advisor_renders_on_submission(self):
        app = self.app()
        app.checkbox[0].check()
        app.selectbox[1].select(OBJECTIVES[2])
        with patch("advisor.OpenAI") as client, patch.dict(os.environ, {"OPENAI_API_KEY": TEST_KEY}):
            create = client.return_value.__enter__.return_value.responses.create
            create.return_value.output_text = '{"action":"yield"}'
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.warning)
        self.assertIn("**التوصية**", [element.value for element in app.markdown])
        self.assertTrue(any("عائد الإيجار" in element.value for element in app.markdown))
        request = create.call_args.kwargs
        self.assertEqual(json.loads(request["input"])["objective"], OBJECTIVES[2])


if __name__ == "__main__":
    unittest.main()
