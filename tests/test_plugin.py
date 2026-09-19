"""Tests for the air_fog plugin.

Everything here exercises ``plugins.air_fog`` — the code this repo ships.
The platform's ``src/utils/air_fog.py`` was a pre-extraction leftover and is
gone; nothing below imports from ``src.utils``.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests as _requests

from plugins.air_fog import (
    OPEN_METEO_AIR_QUALITY_URL,
    PURPLEAIR_SENSORS_URL,
    AirFogPlugin,
    Plugin,
)

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "manifest.json"

# The plugin falls back to these when no location is configured.
SF_LAT = 37.7749
SF_LON = -122.4194


def _manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _response(payload):
    resp = Mock()
    resp.status_code = 200
    resp.json.return_value = payload
    return resp


def _purpleair_sensor(pm25):
    return {"sensor": {"sensor_index": 12345, "stats": {"pm2.5_10minute": pm25}}}


def _owm(visibility_m, humidity, temp_f):
    return {"visibility": visibility_m, "main": {"humidity": humidity, "temp": temp_f}}


def _pollen(grass=0, birch=0, alder=0, ragweed=0, mugwort=0, olive=0):
    return {
        "current": {
            "grass_pollen": grass,
            "birch_pollen": birch,
            "alder_pollen": alder,
            "ragweed_pollen": ragweed,
            "mugwort_pollen": mugwort,
            "olive_pollen": olive,
        }
    }


def _route_by_url(purpleair=None, owm=None, pollen=None):
    """A ``requests.get`` side effect that answers each upstream by URL."""

    def side_effect(url, **kwargs):
        if "purpleair" in url:
            if purpleair is None:
                raise Exception("no PurpleAir response scripted")
            return _response(purpleair)
        if "openweathermap" in url:
            if owm is None:
                raise Exception("no OpenWeatherMap response scripted")
            return _response(owm)
        if "open-meteo" in url:
            if pollen is None:
                raise Exception("no Open-Meteo response scripted")
            return _response(pollen)
        raise AssertionError(f"unexpected URL requested: {url}")

    return side_effect


@pytest.fixture
def plugin():
    return AirFogPlugin(_manifest())


class TestPluginConstruction:
    """What the platform does with this package: import ``Plugin`` and build it."""

    def test_module_exports_the_plugin_class(self):
        assert Plugin is AirFogPlugin

    def test_plugin_id_matches_manifest(self, plugin):
        assert plugin.plugin_id == _manifest()["id"] == "air_fog"

    def test_validate_config_accepts_purpleair_key_alone(self, plugin):
        assert plugin.validate_config({"purpleair_api_key": "k"}) == []

    def test_validate_config_accepts_openweathermap_key_alone(self, plugin):
        assert plugin.validate_config({"openweathermap_api_key": "k"}) == []

    def test_validate_config_rejects_no_keys(self, plugin):
        errors = plugin.validate_config({})
        assert len(errors) == 1
        assert "API key" in errors[0]


class TestDewPointCalculation:
    """Dew point via the Magnus formula — the core fog-prediction input."""

    def test_dew_point_at_100_percent_humidity_equals_temperature(self):
        assert abs(AirFogPlugin.calculate_dew_point(68.0, 100.0) - 68.0) < 0.5

    def test_dew_point_at_50_percent_humidity(self):
        dew_point = AirFogPlugin.calculate_dew_point(70.0, 50.0)
        assert dew_point < 70.0
        assert 48 < dew_point < 52

    def test_dew_point_at_low_humidity(self):
        assert AirFogPlugin.calculate_dew_point(80.0, 20.0) < 40

    def test_dew_point_cold_conditions(self):
        dew_point = AirFogPlugin.calculate_dew_point(32.0, 80.0)
        assert 20 < dew_point < 32.0

    def test_dew_point_hot_conditions(self):
        dew_point = AirFogPlugin.calculate_dew_point(100.0, 70.0)
        assert 85 < dew_point < 92

    def test_dew_point_fog_condition(self):
        assert 55.0 - AirFogPlugin.calculate_dew_point(55.0, 95.0) < 3

    def test_dew_point_returns_float(self):
        assert isinstance(AirFogPlugin.calculate_dew_point(70.0, 60.0), float)

    def test_dew_point_typical_marine_layer(self):
        assert 58.0 - AirFogPlugin.calculate_dew_point(58.0, 92.0) < 5

    def test_dew_point_extreme_cold(self):
        assert AirFogPlugin.calculate_dew_point(-10.0, 60.0) < -10.0

    def test_dew_point_extreme_heat(self):
        assert AirFogPlugin.calculate_dew_point(115.0, 30.0) < 80

    def test_dew_point_near_100_humidity(self):
        assert abs(AirFogPlugin.calculate_dew_point(72.0, 99.0) - 72.0) < 1.0

    def test_dew_point_very_low_humidity(self):
        assert AirFogPlugin.calculate_dew_point(70.0, 10.0) < 20


class TestAQICalculation:
    """AQI from PM2.5 using the US EPA breakpoint table."""

    @pytest.mark.parametrize(
        "pm25,lo,hi,category,color",
        [
            (5.0, 0, 50, "GOOD", "GREEN"),
            (20.0, 51, 100, "MODERATE", "YELLOW"),
            (40.0, 101, 150, "UNHEALTHY_SENSITIVE", "ORANGE"),
            (100.0, 151, 200, "UNHEALTHY", "RED"),
            (200.0, 201, 300, "VERY_UNHEALTHY", "PURPLE"),
            (300.0, 301, 500, "HAZARDOUS", "MAROON"),
        ],
    )
    def test_each_band(self, pm25, lo, hi, category, color):
        aqi, got_category, got_color = AirFogPlugin.calculate_aqi_from_pm25(pm25)
        assert lo <= aqi <= hi
        assert got_category == category
        assert got_color == color

    def test_extreme_pm25_caps_at_500(self):
        assert AirFogPlugin.calculate_aqi_from_pm25(600.0) == (500, "HAZARDOUS", "MAROON")

    def test_zero_pm25(self):
        assert AirFogPlugin.calculate_aqi_from_pm25(0.0) == (0, "GOOD", "GREEN")

    def test_negative_pm25_treated_as_zero(self):
        assert AirFogPlugin.calculate_aqi_from_pm25(-5.0) == (0, "GOOD", "GREEN")

    def test_fire_trigger_threshold(self):
        aqi, _, _ = AirFogPlugin.calculate_aqi_from_pm25(35.0)
        assert aqi <= 100
        aqi, _, _ = AirFogPlugin.calculate_aqi_from_pm25(36.0)
        assert aqi > 100

    def test_aqi_breakpoint_boundaries(self):
        assert AirFogPlugin.calculate_aqi_from_pm25(12.0)[:2] == (50, "GOOD")
        assert AirFogPlugin.calculate_aqi_from_pm25(12.1)[:2] == (51, "MODERATE")

    @pytest.mark.parametrize(
        "pm25,expected_aqi,expected_category",
        [
            (12.05, 50, "GOOD"),
            (35.45, 100, "MODERATE"),
            (55.45, 150, "UNHEALTHY_SENSITIVE"),
            (150.45, 200, "UNHEALTHY"),
            (250.45, 300, "VERY_UNHEALTHY"),
        ],
    )
    def test_values_between_breakpoints_do_not_fall_through(
        self, pm25, expected_aqi, expected_category
    ):
        """Regression: values in the 0.1-wide gaps between bands must not become 500.

        PM2.5 is averaged across several sensors so it lands on arbitrary
        floats. EPA truncates to one decimal before the lookup, which closes
        the gaps.
        """
        aqi, category, _ = AirFogPlugin.calculate_aqi_from_pm25(pm25)
        assert aqi == expected_aqi
        assert category == expected_category


class TestFogStatus:
    """``determine_fog_status`` returns (is_foggy, status, colour)."""

    def test_fog_when_visibility_below_1600m(self, plugin):
        assert plugin.determine_fog_status(1000, 70, 65) == (True, "FOG", "ORANGE")

    def test_visibility_threshold_is_exclusive(self, plugin):
        assert plugin.determine_fog_status(1599, 70, 65)[0] is True
        assert plugin.determine_fog_status(1600, 70, 65)[0] is False

    def test_fog_when_humid_and_cold(self, plugin):
        assert plugin.determine_fog_status(5000, 96, 55) == (True, "FOG", "ORANGE")

    def test_no_fog_when_humid_but_not_cold(self, plugin):
        assert plugin.determine_fog_status(5000, 96, 60)[0] is False

    def test_no_fog_when_cold_but_not_humid_enough(self, plugin):
        assert plugin.determine_fog_status(5000, 95, 55)[0] is False

    def test_haze_between_1600m_and_3000m(self, plugin):
        assert plugin.determine_fog_status(2500, 70, 65) == (False, "HAZE", "YELLOW")

    def test_clear_at_3000m_and_above(self, plugin):
        assert plugin.determine_fog_status(10000, 50, 70) == (False, "CLEAR", "GREEN")

    def test_visibility_wins_over_dry_warm_air(self, plugin):
        assert plugin.determine_fog_status(500, 30, 80) == (True, "FOG", "ORANGE")


class TestAirStatus:
    """``determine_air_status`` maps an AQI to a status word and colour."""

    @pytest.mark.parametrize(
        "aqi,expected",
        [
            (40, ("GOOD", "GREEN")),
            (75, ("MODERATE", "YELLOW")),
            (125, ("MODERATE HIGH", "ORANGE")),
            (175, ("UNHEALTHY", "RED")),
            (250, ("VERY UNHEALTHY", "PURPLE")),
            (350, ("HAZARDOUS", "MAROON")),
        ],
    )
    def test_each_band(self, plugin, aqi, expected):
        assert plugin.determine_air_status(aqi) == expected

    def test_fire_trigger_at_aqi_100_boundary(self, plugin):
        assert plugin.determine_air_status(100) == ("MODERATE", "YELLOW")
        assert plugin.determine_air_status(101) == ("MODERATE HIGH", "ORANGE")


class TestPollenLevel:
    """``determine_pollen_level`` against each species' threshold table."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (10, ("LOW", "GREEN")),
            (50, ("MODERATE", "YELLOW")),
            (100, ("HIGH", "ORANGE")),
            (300, ("VERY HIGH", "RED")),
            (0, ("LOW", "GREEN")),
            (-5, ("LOW", "GREEN")),
        ],
    )
    def test_grass_bands(self, value, expected):
        assert (
            AirFogPlugin.determine_pollen_level(value, AirFogPlugin.GRASS_POLLEN_THRESHOLDS)
            == expected
        )

    @pytest.mark.parametrize(
        "value,expected",
        [
            (30, ("LOW", "GREEN")),
            (100, ("MODERATE", "YELLOW")),
            (500, ("HIGH", "ORANGE")),
            (800, ("VERY HIGH", "RED")),
        ],
    )
    def test_tree_bands(self, value, expected):
        assert (
            AirFogPlugin.determine_pollen_level(value, AirFogPlugin.TREE_POLLEN_THRESHOLDS)
            == expected
        )

    @pytest.mark.parametrize(
        "value,expected",
        [
            (5, ("LOW", "GREEN")),
            (50, ("MODERATE", "YELLOW")),
            (200, ("HIGH", "ORANGE")),
        ],
    )
    def test_weed_bands(self, value, expected):
        assert (
            AirFogPlugin.determine_pollen_level(value, AirFogPlugin.WEED_POLLEN_THRESHOLDS)
            == expected
        )

    def test_grass_low_moderate_boundary(self):
        table = AirFogPlugin.GRASS_POLLEN_THRESHOLDS
        assert AirFogPlugin.determine_pollen_level(20, table)[0] == "LOW"
        assert AirFogPlugin.determine_pollen_level(21, table)[0] == "MODERATE"

    def test_grass_moderate_high_boundary(self):
        table = AirFogPlugin.GRASS_POLLEN_THRESHOLDS
        assert AirFogPlugin.determine_pollen_level(77, table)[0] == "MODERATE"
        assert AirFogPlugin.determine_pollen_level(78, table)[0] == "HIGH"


class TestColorCodes:
    def test_color_to_code(self, plugin):
        assert plugin._color_to_code("GREEN") == 66
        assert plugin._color_to_code("YELLOW") == 65
        assert plugin._color_to_code("ORANGE") == 64
        assert plugin._color_to_code("RED") == 63
        assert plugin._color_to_code("PURPLE") == 68
        assert plugin._color_to_code("MAROON") == 68

    def test_unknown_color_falls_back_to_green(self, plugin):
        assert plugin._color_to_code("UNKNOWN") == 66


class TestLocation:
    """Where the plugin asks for data when a location is (not) configured."""

    def test_default_location_is_san_francisco(self, plugin):
        plugin.config = {"openweathermap_api_key": "k"}
        with patch("plugins.air_fog.requests.get", return_value=_response(_owm(1, 1, 1))) as get:
            plugin._fetch_openweathermap_data()
        params = get.call_args.kwargs["params"]
        assert (params["lat"], params["lon"]) == (SF_LAT, SF_LON)

    def test_configured_location_is_sent_to_every_upstream(self, plugin):
        plugin.config = {
            "purpleair_api_key": "k",
            "openweathermap_api_key": "k",
            "latitude": 34.0,
            "longitude": -118.0,
        }
        side_effect = _route_by_url(
            purpleair={"fields": ["sensor_index", "pm2.5_10minute"], "data": [[1, 10.0]]},
            owm=_owm(10000, 50, 70),
            pollen=_pollen(),
        )
        with patch("plugins.air_fog.requests.get", side_effect=side_effect) as get:
            plugin.fetch_data()

        by_url = {call.args[0]: call.kwargs["params"] for call in get.call_args_list}
        assert by_url[PURPLEAIR_SENSORS_URL]["nwlat"] == pytest.approx(34.0 + 0.05)
        assert by_url[PURPLEAIR_SENSORS_URL]["selng"] == pytest.approx(-118.0 + 0.05)
        owm_url = next(u for u in by_url if "openweathermap" in u)
        assert (by_url[owm_url]["lat"], by_url[owm_url]["lon"]) == (34.0, -118.0)
        assert by_url[OPEN_METEO_AIR_QUALITY_URL]["latitude"] == 34.0
        assert by_url[OPEN_METEO_AIR_QUALITY_URL]["longitude"] == -118.0


class TestPurpleAir:
    """``_fetch_purpleair_data``: single-sensor and nearby-sensor modes."""

    def test_sensor_id_reads_from_stats_object(self, plugin):
        """GET /v1/sensors/:id exposes running averages through ``stats``."""
        plugin.config = {"purpleair_api_key": "test_key", "purpleair_sensor_id": "12345"}
        payload = {"sensor": {"sensor_index": 12345, "stats": {"pm2.5": 24.0, "pm2.5_10minute": 25.5}}}
        with patch("plugins.air_fog.requests.get", return_value=_response(payload)):
            result = plugin._fetch_purpleair_data()
        assert result["pm2_5"] == 25.5
        assert result["aqi_category"] == "MODERATE"

    def test_sensor_id_falls_back_to_top_level_reading(self, plugin):
        plugin.config = {"purpleair_api_key": "test_key", "purpleair_sensor_id": "12345"}
        payload = {"sensor": {"sensor_index": 12345, "pm2.5": 25.5}}
        with patch("plugins.air_fog.requests.get", return_value=_response(payload)):
            assert plugin._fetch_purpleair_data()["pm2_5"] == 25.5

    def test_sensor_id_requests_the_sensor_endpoint(self, plugin):
        plugin.config = {"purpleair_api_key": "test_key", "purpleair_sensor_id": "12345"}
        with patch(
            "plugins.air_fog.requests.get", return_value=_response(_purpleair_sensor(10.0))
        ) as get:
            plugin._fetch_purpleair_data()
        assert get.call_args.args[0] == f"{PURPLEAIR_SENSORS_URL}/12345"
        assert get.call_args.kwargs["headers"] == {"X-API-Key": "test_key"}

    def test_sensor_with_no_reading_yields_no_data(self, plugin):
        """No PM2.5 value must not become AQI 0."""
        plugin.config = {"purpleair_api_key": "test_key", "purpleair_sensor_id": "12345"}
        payload = {"sensor": {"sensor_index": 12345}}
        with patch("plugins.air_fog.requests.get", return_value=_response(payload)):
            assert plugin._fetch_purpleair_data() is None

    def test_nearby_sensors_are_averaged(self, plugin):
        """PurpleAir prepends sensor_index, so PM2.5 is column 1 here."""
        plugin.config = {"purpleair_api_key": "test_key"}
        payload = {
            "fields": ["sensor_index", "pm2.5_10minute"],
            "data": [[77245, 30.0], [95189, 35.0], [142936, 28.0]],
        }
        with patch("plugins.air_fog.requests.get", return_value=_response(payload)):
            assert plugin._fetch_purpleair_data()["pm2_5"] == 31.0

    def test_nearby_column_is_resolved_by_name(self, plugin):
        """The API docs warn column order may change; parse by the fields array."""
        plugin.config = {"purpleair_api_key": "test_key"}
        payload = {
            "fields": ["sensor_index", "name", "humidity", "pm2.5_10minute", "temperature"],
            "data": [[77245, "A", 48, 30.0, 71], [95189, "B", 51, 32.0, 69]],
        }
        with patch("plugins.air_fog.requests.get", return_value=_response(payload)):
            assert plugin._fetch_purpleair_data()["pm2_5"] == 31.0

    def test_nearby_missing_pm25_column_yields_no_data(self, plugin):
        plugin.config = {"purpleair_api_key": "test_key"}
        payload = {"fields": ["sensor_index", "humidity"], "data": [[77245, 48]]}
        with patch("plugins.air_fog.requests.get", return_value=_response(payload)):
            assert plugin._fetch_purpleair_data() is None

    def test_implausible_reading_is_rejected(self, plugin):
        """Regression: sensor indexes parsed as PM2.5 must not become AQI 500."""
        plugin.config = {"purpleair_api_key": "test_key"}
        payload = {"fields": ["pm2.5_10minute"], "data": [[77245], [95189], [142936]]}
        with patch("plugins.air_fog.requests.get", return_value=_response(payload)):
            assert plugin._fetch_purpleair_data() is None

    def test_no_api_key_makes_no_request(self, plugin):
        plugin.config = {}
        with patch("plugins.air_fog.requests.get") as get:
            assert plugin._fetch_purpleair_data() is None
        get.assert_not_called()

    def test_network_error_is_swallowed(self, plugin):
        plugin.config = {"purpleair_api_key": "test_key", "purpleair_sensor_id": "12345"}
        with patch("plugins.air_fog.requests.get", side_effect=Exception("API error")):
            assert plugin._fetch_purpleair_data() is None

    def test_http_error_is_swallowed(self, plugin):
        """A rejected API key (HTTP 403) is handled, not raised."""
        plugin.config = {"purpleair_api_key": "bad_key", "purpleair_sensor_id": "12345"}
        error_resp = Mock()
        error_resp.status_code = 403
        error_resp.text = '{"error": "ApiKeyInvalidError"}'
        with patch(
            "plugins.air_fog.requests.get",
            side_effect=_requests.HTTPError(response=error_resp),
        ):
            assert plugin._fetch_purpleair_data() is None

    def test_read_key_is_forwarded_for_private_sensors(self, plugin):
        plugin.config = {
            "purpleair_api_key": "test_key",
            "purpleair_sensor_id": "12345",
            "purpleair_read_key": "sensor_read_key",
        }
        with patch(
            "plugins.air_fog.requests.get", return_value=_response(_purpleair_sensor(10.0))
        ) as get:
            plugin._fetch_purpleair_data()
        assert get.call_args.kwargs["params"]["read_key"] == "sensor_read_key"

    def test_no_nearby_sensors_yields_no_data(self, plugin):
        plugin.config = {"purpleair_api_key": "test_key"}
        with patch("plugins.air_fog.requests.get", return_value=_response({"data": []})):
            assert plugin._fetch_purpleair_data() is None

    def test_all_null_nearby_readings_yield_no_data(self, plugin):
        plugin.config = {"purpleair_api_key": "test_key"}
        payload = {"fields": ["sensor_index", "pm2.5_10minute"], "data": [[1, None], [2, None]]}
        with patch("plugins.air_fog.requests.get", return_value=_response(payload)):
            assert plugin._fetch_purpleair_data() is None


class TestOpenWeatherMap:
    def test_success_returns_visibility_humidity_temperature(self, plugin):
        plugin.config = {"openweathermap_api_key": "test_key"}
        with patch("plugins.air_fog.requests.get", return_value=_response(_owm(5000, 75, 62.5))):
            result = plugin._fetch_openweathermap_data()
        assert result == {"visibility_m": 5000, "humidity": 75, "temperature_f": 62.5}

    def test_request_asks_for_imperial_units(self, plugin):
        plugin.config = {"openweathermap_api_key": "test_key"}
        with patch("plugins.air_fog.requests.get", return_value=_response(_owm(1, 1, 1))) as get:
            plugin._fetch_openweathermap_data()
        params = get.call_args.kwargs["params"]
        assert params["appid"] == "test_key"
        assert params["units"] == "imperial"

    def test_no_api_key_makes_no_request(self, plugin):
        plugin.config = {}
        with patch("plugins.air_fog.requests.get") as get:
            assert plugin._fetch_openweathermap_data() is None
        get.assert_not_called()

    def test_network_error_is_swallowed(self, plugin):
        plugin.config = {"openweathermap_api_key": "test_key"}
        with patch("plugins.air_fog.requests.get", side_effect=Exception("Network error")):
            assert plugin._fetch_openweathermap_data() is None


class TestPollenFetch:
    """``_fetch_pollen_data`` against Open-Meteo (no key required)."""

    def test_success_sums_tree_and_weed_species(self, plugin):
        payload = _pollen(grass=15.0, birch=40.0, alder=25.0, ragweed=10.0, mugwort=5.0, olive=20.0)
        with patch("plugins.air_fog.requests.get", return_value=_response(payload)):
            result = plugin._fetch_pollen_data()
        assert result["grass_pollen"] == 15.0
        assert result["tree_pollen"] == 85.0  # birch + alder + olive
        assert result["weed_pollen"] == 15.0  # ragweed + mugwort
        assert result["grass_pollen_level"] == "LOW"
        assert result["tree_pollen_level"] == "MODERATE"
        assert result["weed_pollen_level"] == "LOW"

    def test_requests_every_species_from_open_meteo(self, plugin):
        with patch("plugins.air_fog.requests.get", return_value=_response(_pollen())) as get:
            plugin._fetch_pollen_data()
        assert get.call_args.args[0] == OPEN_METEO_AIR_QUALITY_URL
        requested = set(get.call_args.kwargs["params"]["current"].split(","))
        assert requested == {
            "grass_pollen",
            "birch_pollen",
            "alder_pollen",
            "ragweed_pollen",
            "mugwort_pollen",
            "olive_pollen",
        }

    def test_null_values_count_as_zero(self, plugin):
        payload = _pollen(grass=None, birch=None, alder=None, ragweed=None, mugwort=None, olive=None)
        with patch("plugins.air_fog.requests.get", return_value=_response(payload)):
            result = plugin._fetch_pollen_data()
        assert (result["grass_pollen"], result["tree_pollen"], result["weed_pollen"]) == (0, 0, 0)
        assert result["grass_pollen_level"] == "LOW"

    def test_empty_current_block_counts_as_zero(self, plugin):
        with patch("plugins.air_fog.requests.get", return_value=_response({"current": {}})):
            result = plugin._fetch_pollen_data()
        assert (result["grass_pollen"], result["tree_pollen"], result["weed_pollen"]) == (0, 0, 0)

    def test_high_values_are_very_high(self, plugin):
        payload = _pollen(grass=300.0, birch=400.0, alder=250.0, ragweed=200.0, mugwort=100.0, olive=150.0)
        with patch("plugins.air_fog.requests.get", return_value=_response(payload)):
            result = plugin._fetch_pollen_data()
        assert result["grass_pollen_level"] == "VERY HIGH"
        assert result["grass_pollen_color"] == "RED"
        assert result["tree_pollen"] == 800.0
        assert result["tree_pollen_level"] == "VERY HIGH"
        assert result["weed_pollen"] == 300.0
        assert result["weed_pollen_level"] == "VERY HIGH"

    def test_network_error_is_swallowed(self, plugin):
        with patch("plugins.air_fog.requests.get", side_effect=Exception("Network error")):
            assert plugin._fetch_pollen_data() is None


class TestFetchData:
    """``fetch_data`` combines the three upstreams into the template payload."""

    def test_all_three_upstreams(self, plugin):
        plugin.config = {
            "purpleair_api_key": "purple_key",
            "openweathermap_api_key": "owm_key",
            "purpleair_sensor_id": "12345",
        }
        side_effect = _route_by_url(
            purpleair=_purpleair_sensor(45.0),  # UNHEALTHY_SENSITIVE
            owm=_owm(1200, 92, 55.0),  # foggy
            pollen=_pollen(grass=5.0, birch=10.0, alder=8.0, ragweed=3.0, mugwort=2.0, olive=7.0),
        )
        with patch("plugins.air_fog.requests.get", side_effect=side_effect):
            result = plugin.fetch_data()

        assert result.available
        data = result.data
        assert 101 <= data["aqi"] <= 150
        assert data["air_status"] == "MODERATE HIGH"
        assert data["air_color"] == "{64}"
        assert data["is_foggy"] == "Yes"
        assert data["fog_status"] == "FOG"
        assert data["fog_color"] == "{64}"
        assert data["visibility"] == "0.7mi"
        assert data["grass_pollen"] == 5.0
        assert data["tree_pollen"] == 25.0
        assert data["weed_pollen"] == 5.0
        assert data["formatted"] == f"AQI:{data['aqi']} VIS:0.7mi GRASS:5.0 TREES:25.0 WEEDS:5.0"

    def test_payload_keys_match_manifest_variables(self, plugin):
        """Every variable the manifest declares is produced, and nothing else."""
        plugin.config = {"purpleair_api_key": "k", "openweathermap_api_key": "k", "purpleair_sensor_id": "1"}
        side_effect = _route_by_url(
            purpleair=_purpleair_sensor(10.0), owm=_owm(10000, 50, 70.0), pollen=_pollen()
        )
        with patch("plugins.air_fog.requests.get", side_effect=side_effect):
            data = plugin.fetch_data().data
        assert set(data) == set(_manifest()["variables"]["simple"])

    def test_pollen_only_when_no_keys_configured(self, plugin):
        """Open-Meteo needs no key, so a keyless config still yields pollen."""
        plugin.config = {}
        side_effect = _route_by_url(pollen=_pollen(grass=50.0))
        with patch("plugins.air_fog.requests.get", side_effect=side_effect) as get:
            result = plugin.fetch_data()

        assert result.available
        assert get.call_count == 1  # only Open-Meteo was asked
        assert result.data["grass_pollen"] == 50.0
        assert result.data["grass_pollen_level"] == "MODERATE"
        assert result.data["aqi"] is None
        assert result.data["air_status"] == "UNKNOWN"
        assert result.data["fog_status"] == "UNKNOWN"
        assert result.data["formatted"] == "GRASS:50.0 TREES:0 WEEDS:0"

    def test_air_and_fog_without_pollen(self, plugin):
        plugin.config = {"purpleair_api_key": "k", "openweathermap_api_key": "k", "purpleair_sensor_id": "1"}
        side_effect = _route_by_url(purpleair=_purpleair_sensor(20.0), owm=_owm(5000, 75, 62.5))
        with patch("plugins.air_fog.requests.get", side_effect=side_effect):
            result = plugin.fetch_data()

        assert result.available
        assert result.data["aqi"] == 68
        assert result.data["air_status"] == "MODERATE"
        assert result.data["visibility"] == "3.1mi"
        assert result.data["is_foggy"] == "No"
        assert result.data["grass_pollen"] is None
        assert result.data["grass_pollen_level"] == "UNKNOWN"
        assert result.data["formatted"] == "AQI:68 VIS:3.1mi"

    def test_every_upstream_failing_is_unavailable(self, plugin):
        plugin.config = {"purpleair_api_key": "k", "openweathermap_api_key": "k"}
        with patch("plugins.air_fog.requests.get", side_effect=Exception("down")):
            result = plugin.fetch_data()
        assert not result.available
        assert result.data is None
        assert result.error == "Failed to fetch data from any source"

    def test_one_failing_upstream_does_not_take_the_others_down(self, plugin):
        plugin.config = {"purpleair_api_key": "k", "openweathermap_api_key": "k", "purpleair_sensor_id": "1"}
        side_effect = _route_by_url(owm=_owm(10000, 50, 70.0), pollen=_pollen())  # PurpleAir raises
        with patch("plugins.air_fog.requests.get", side_effect=side_effect):
            result = plugin.fetch_data()
        assert result.available
        assert result.data["aqi"] is None
        assert result.data["fog_status"] == "CLEAR"
        assert result.data["fog_color"] == "{66}"


MANIFEST_REQUIRED_VAR_FIELDS = {"description", "type", "max_length", "group", "example"}

EXPECTED_SIMPLE_VARS = [
    "aqi", "air_status", "air_color",
    "fog_status", "fog_color", "is_foggy", "visibility",
    "grass_pollen", "grass_pollen_level", "grass_pollen_color",
    "tree_pollen", "tree_pollen_level", "tree_pollen_color",
    "weed_pollen", "weed_pollen_level", "weed_pollen_color",
    "formatted",
]


class TestManifestMetadata:
    """Validate the rich variable metadata in manifest.json."""

    @pytest.fixture(autouse=True)
    def load_manifest(self):
        self.manifest = _manifest()
        self.variables = self.manifest["variables"]
        self.simple = self.variables["simple"]
        self.groups = self.variables["groups"]

    def test_required_top_level_fields(self):
        for field in ("id", "name", "version"):
            assert field in self.manifest

    def test_simple_is_dict(self):
        assert isinstance(self.simple, dict), "variables.simple must be a dict, not a list"

    def test_expected_variable_count(self):
        assert len(self.simple) == 17

    def test_all_expected_vars_present(self):
        assert set(self.simple.keys()) == set(EXPECTED_SIMPLE_VARS)

    def test_each_variable_has_required_fields(self):
        for var_name, meta in self.simple.items():
            missing = MANIFEST_REQUIRED_VAR_FIELDS - set(meta.keys())
            assert not missing, f"{var_name} missing fields: {missing}"

    def test_groups_section_exists(self):
        assert isinstance(self.groups, dict)
        assert len(self.groups) >= 1

    def test_every_variable_references_valid_group(self):
        for var_name, meta in self.simple.items():
            assert meta["group"] in self.groups, (
                f"{var_name} references unknown group '{meta['group']}'"
            )

    def test_max_length_is_positive_int(self):
        for var_name, meta in self.simple.items():
            ml = meta["max_length"]
            assert isinstance(ml, int) and ml > 0, (
                f"{var_name}: max_length must be a positive int, got {ml}"
            )

    def test_type_values_are_valid(self):
        allowed = {"string", "number", "boolean"}
        for var_name, meta in self.simple.items():
            assert meta["type"] in allowed, f"{var_name}: invalid type '{meta['type']}'"

    def test_no_old_max_lengths_key(self):
        assert "max_lengths" not in self.manifest, (
            "Old top-level max_lengths key should be removed"
        )
