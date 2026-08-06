"""Air Quality, Fog & Allergen plugin for FiestaBoard.

Displays air quality (AQI), fog/visibility conditions, and pollen/allergen levels.
"""

from typing import Any, Dict, List, Optional
import logging
import math
import requests

from src.plugins.base import PluginBase, PluginResult

logger = logging.getLogger(__name__)

# Open-Meteo Air Quality API (free, no key required)
OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# PurpleAir API
PURPLEAIR_SENSORS_URL = "https://api.purpleair.com/v1/sensors"
PURPLEAIR_PM25_FIELD = "pm2.5_10minute"
PURPLEAIR_PM25_FALLBACK_FIELD = "pm2.5"
PURPLEAIR_BBOX_DEGREES = 0.05
PURPLEAIR_MAX_AGE_SECONDS = 3600

# A consumer sensor reading above this is a parsing error, not weather. Guards
# against silently reporting a fabricated AQI if the response shape changes.
MAX_PLAUSIBLE_PM25 = 1000.0


class AirFogPlugin(PluginBase):
    """Air Quality, Fog, and Allergen data plugin.
    
    Fetches AQI from PurpleAir, visibility from OpenWeatherMap,
    and pollen data from Open-Meteo.
    """
    
    # AQI breakpoints for PM2.5 (US EPA standard)
    AQI_BREAKPOINTS = [
        (0.0, 12.0, 0, 50, "GOOD", "GREEN"),
        (12.1, 35.4, 51, 100, "MODERATE", "YELLOW"),
        (35.5, 55.4, 101, 150, "UNHEALTHY_SENSITIVE", "ORANGE"),
        (55.5, 150.4, 151, 200, "UNHEALTHY", "RED"),
        (150.5, 250.4, 201, 300, "VERY_UNHEALTHY", "PURPLE"),
        (250.5, 500.4, 301, 500, "HAZARDOUS", "MAROON"),
    ]
    
    # Pollen severity thresholds (grains/m³)
    GRASS_POLLEN_THRESHOLDS = [
        (0, 20, "LOW", "GREEN"),
        (21, 77, "MODERATE", "YELLOW"),
        (78, 266, "HIGH", "ORANGE"),
        (267, float("inf"), "VERY HIGH", "RED"),
    ]
    TREE_POLLEN_THRESHOLDS = [
        (0, 50, "LOW", "GREEN"),
        (51, 200, "MODERATE", "YELLOW"),
        (201, 700, "HIGH", "ORANGE"),
        (701, float("inf"), "VERY HIGH", "RED"),
    ]
    WEED_POLLEN_THRESHOLDS = [
        (0, 20, "LOW", "GREEN"),
        (21, 77, "MODERATE", "YELLOW"),
        (78, 266, "HIGH", "ORANGE"),
        (267, float("inf"), "VERY HIGH", "RED"),
    ]
    
    # Thresholds
    VISIBILITY_FOG_THRESHOLD_M = 1600
    HUMIDITY_FOG_THRESHOLD = 95
    TEMP_FOG_THRESHOLD_F = 60
    AQI_FIRE_THRESHOLD = 100
    
    def __init__(self, manifest: Dict[str, Any]):
        """Initialize the air/fog plugin."""
        super().__init__(manifest)
    
    @property
    def plugin_id(self) -> str:
        return "air_fog"
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate air/fog configuration."""
        errors = []
        
        purpleair_key = config.get("purpleair_api_key")
        owm_key = config.get("openweathermap_api_key")
        
        if not purpleair_key and not owm_key:
            errors.append("At least one API key (PurpleAir or OpenWeatherMap) is required")
        
        return errors
    
    @staticmethod
    def calculate_dew_point(temp_f: float, humidity: float) -> float:
        """Calculate dew point using Magnus formula."""
        temp_c = (temp_f - 32) * 5 / 9
        a, b = 17.27, 237.7
        alpha = (a * temp_c / (b + temp_c)) + math.log(humidity / 100)
        dew_point_c = (b * alpha) / (a - alpha)
        return round((dew_point_c * 9 / 5) + 32, 1)
    
    @staticmethod
    def calculate_aqi_from_pm25(pm25: float) -> tuple:
        """Calculate AQI from PM2.5 concentration."""
        if pm25 < 0:
            pm25 = 0

        # EPA truncates PM2.5 to one decimal place before applying the table.
        # Without this, an averaged reading landing between two breakpoints
        # (e.g. 35.45) matches no band and falls through to the 500 fallback.
        pm25 = math.floor(pm25 * 10) / 10

        for bp_low, bp_high, aqi_low, aqi_high, category, color in AirFogPlugin.AQI_BREAKPOINTS:
            if bp_low <= pm25 <= bp_high:
                aqi = round(
                    ((aqi_high - aqi_low) / (bp_high - bp_low)) * (pm25 - bp_low) + aqi_low
                )
                return aqi, category, color
        
        return 500, "HAZARDOUS", "MAROON"
    
    @staticmethod
    def determine_pollen_level(value: float, thresholds: list) -> tuple:
        """Determine pollen severity level from concentration."""
        if value < 0:
            value = 0
        for low, high, level, color in thresholds:
            if low <= value <= high:
                return level, color
        return "VERY HIGH", "RED"
    
    def determine_fog_status(self, visibility_m: float, humidity: float, temp_f: float) -> tuple:
        """Determine fog status based on conditions."""
        if visibility_m < self.VISIBILITY_FOG_THRESHOLD_M:
            return True, "FOG", "ORANGE"
        if humidity > self.HUMIDITY_FOG_THRESHOLD and temp_f < self.TEMP_FOG_THRESHOLD_F:
            return True, "FOG", "ORANGE"
        if visibility_m < 3000:
            return False, "HAZE", "YELLOW"
        return False, "CLEAR", "GREEN"
    
    def determine_air_status(self, aqi: int) -> tuple:
        """Determine air quality status."""
        if aqi > 300:
            return "HAZARDOUS", "MAROON"
        elif aqi > 200:
            return "VERY UNHEALTHY", "PURPLE"
        elif aqi > 150:
            return "UNHEALTHY", "RED"
        elif aqi > self.AQI_FIRE_THRESHOLD:
            return "MODERATE HIGH", "ORANGE"
        elif aqi > 50:
            return "MODERATE", "YELLOW"
        else:
            return "GOOD", "GREEN"
    
    @staticmethod
    def _pm25_column(fields: List[str]) -> Optional[int]:
        """Find the PM2.5 column in a PurpleAir ``fields`` array.

        PurpleAir prepends ``sensor_index`` to whatever fields you request, and
        the API docs warn that column order may change as columns are added, so
        the column is always resolved by name rather than by position.
        """
        for name in (PURPLEAIR_PM25_FIELD, PURPLEAIR_PM25_FALLBACK_FIELD):
            if name in fields:
                return fields.index(name)
        return None

    def _fetch_single_sensor_pm25(
        self, sensor_id: str, read_key: Optional[str], headers: Dict[str, str]
    ) -> Optional[float]:
        """Fetch PM2.5 for one specific sensor."""
        params = {}
        if read_key:
            params["read_key"] = read_key

        # ``fields`` is optional on this endpoint, and the running averages are
        # only exposed through the stats object here, so request the default
        # payload rather than asking for a field name this endpoint may reject.
        response = requests.get(
            f"{PURPLEAIR_SENSORS_URL}/{sensor_id}",
            params=params,
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        sensor = response.json().get("sensor") or {}

        for source in (sensor.get("stats") or {}, sensor):
            for field in (PURPLEAIR_PM25_FIELD, PURPLEAIR_PM25_FALLBACK_FIELD):
                value = source.get(field)
                if value is not None:
                    return float(value)

        logger.warning("PurpleAir sensor %s returned no PM2.5 reading", sensor_id)
        return None

    def _fetch_nearby_sensors_pm25(
        self, lat: float, lon: float, read_key: Optional[str], headers: Dict[str, str]
    ) -> Optional[float]:
        """Fetch PM2.5 averaged across sensors in a box around lat/lon."""
        params = {
            "fields": PURPLEAIR_PM25_FIELD,
            "location_type": 0,
            "max_age": PURPLEAIR_MAX_AGE_SECONDS,
            "nwlat": lat + PURPLEAIR_BBOX_DEGREES,
            "nwlng": lon - PURPLEAIR_BBOX_DEGREES,
            "selat": lat - PURPLEAIR_BBOX_DEGREES,
            "selng": lon + PURPLEAIR_BBOX_DEGREES,
        }
        if read_key:
            params["read_keys"] = read_key

        response = requests.get(
            PURPLEAIR_SENSORS_URL, params=params, headers=headers, timeout=10
        )
        response.raise_for_status()
        payload = response.json()

        rows = payload.get("data") or []
        if not rows:
            logger.warning("No PurpleAir sensors found near %s,%s", lat, lon)
            return None

        fields = payload.get("fields") or []
        column = self._pm25_column(fields)
        if column is None:
            logger.error("PurpleAir response has no PM2.5 column; fields=%s", fields)
            return None

        values = [
            row[column]
            for row in rows
            if len(row) > column and row[column] is not None
        ]
        if not values:
            logger.warning("PurpleAir returned %d sensors with no PM2.5 data", len(rows))
            return None

        return sum(values) / len(values)

    def _fetch_purpleair_data(self) -> Optional[Dict]:
        """Fetch air quality from PurpleAir."""
        api_key = self.config.get("purpleair_api_key")
        if not api_key:
            return None

        sensor_id = self.config.get("purpleair_sensor_id")
        read_key = self.config.get("purpleair_read_key")
        lat = self.config.get("latitude", 37.7749)
        lon = self.config.get("longitude", -122.4194)
        headers = {"X-API-Key": api_key}

        try:
            if sensor_id:
                pm25 = self._fetch_single_sensor_pm25(sensor_id, read_key, headers)
            else:
                pm25 = self._fetch_nearby_sensors_pm25(lat, lon, read_key, headers)
        except requests.HTTPError as e:
            response = e.response
            status = response.status_code if response is not None else "unknown"
            body = response.text[:200] if response is not None else ""
            logger.error("PurpleAir request failed with HTTP %s: %s", status, body)
            return None
        except Exception as e:
            logger.error(f"Failed to fetch PurpleAir data: {e}")
            return None

        if pm25 is None:
            return None

        if pm25 > MAX_PLAUSIBLE_PM25:
            logger.error(
                "Discarding implausible PurpleAir PM2.5 reading of %s; "
                "the response shape has likely changed",
                pm25,
            )
            return None

        aqi, category, color = self.calculate_aqi_from_pm25(pm25)

        return {
            "pm2_5": round(pm25, 1),
            "aqi": aqi,
            "aqi_category": category,
            "aqi_color": color,
        }
    
    def _fetch_openweathermap_data(self) -> Optional[Dict]:
        """Fetch visibility from OpenWeatherMap."""
        api_key = self.config.get("openweathermap_api_key")
        if not api_key:
            return None
        
        lat = self.config.get("latitude", 37.7749)
        lon = self.config.get("longitude", -122.4194)
        
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": api_key,
                "units": "imperial"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            visibility_m = data.get("visibility", 10000)
            humidity = data["main"]["humidity"]
            temp_f = data["main"]["temp"]
            
            return {
                "visibility_m": visibility_m,
                "humidity": humidity,
                "temperature_f": round(temp_f, 1),
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch OpenWeatherMap data: {e}")
            return None
    
    def _fetch_pollen_data(self) -> Optional[Dict]:
        """Fetch pollen/allergen data from Open-Meteo Air Quality API (free, no key)."""
        lat = self.config.get("latitude", 37.7749)
        lon = self.config.get("longitude", -122.4194)
        
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "grass_pollen,birch_pollen,alder_pollen,ragweed_pollen,mugwort_pollen,olive_pollen",
            }
            
            response = requests.get(OPEN_METEO_AIR_QUALITY_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            current = data.get("current", {})
            
            grass = current.get("grass_pollen") or 0
            birch = current.get("birch_pollen") or 0
            alder = current.get("alder_pollen") or 0
            ragweed = current.get("ragweed_pollen") or 0
            mugwort = current.get("mugwort_pollen") or 0
            olive = current.get("olive_pollen") or 0
            
            tree_total = birch + alder + olive
            weed_total = ragweed + mugwort
            
            grass_level, grass_color = self.determine_pollen_level(
                grass, self.GRASS_POLLEN_THRESHOLDS
            )
            tree_level, tree_color = self.determine_pollen_level(
                tree_total, self.TREE_POLLEN_THRESHOLDS
            )
            weed_level, weed_color = self.determine_pollen_level(
                weed_total, self.WEED_POLLEN_THRESHOLDS
            )
            
            return {
                "grass_pollen": round(grass, 1),
                "grass_pollen_level": grass_level,
                "grass_pollen_color": grass_color,
                "tree_pollen": round(tree_total, 1),
                "tree_pollen_level": tree_level,
                "tree_pollen_color": tree_color,
                "weed_pollen": round(weed_total, 1),
                "weed_pollen_level": weed_level,
                "weed_pollen_color": weed_color,
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch pollen data: {e}")
            return None
    
    def fetch_data(self) -> PluginResult:
        """Fetch combined air quality, fog, and pollen data."""
        purpleair_data = self._fetch_purpleair_data()
        owm_data = self._fetch_openweathermap_data()
        pollen_data = self._fetch_pollen_data()
        
        if not purpleair_data and not owm_data and not pollen_data:
            return PluginResult(
                available=False,
                error="Failed to fetch data from any source"
            )
        
        result = {
            "aqi": None,
            "air_status": "UNKNOWN",
            "air_color": "",
            "fog_status": "UNKNOWN",
            "fog_color": "",
            "is_foggy": False,
            "visibility": None,
            "grass_pollen": None,
            "grass_pollen_level": "UNKNOWN",
            "grass_pollen_color": "",
            "tree_pollen": None,
            "tree_pollen_level": "UNKNOWN",
            "tree_pollen_color": "",
            "weed_pollen": None,
            "weed_pollen_level": "UNKNOWN",
            "weed_pollen_color": "",
            "formatted": "NO DATA",
        }
        
        if purpleair_data:
            result["aqi"] = purpleair_data["aqi"]
            air_status, air_color = self.determine_air_status(purpleair_data["aqi"])
            result["air_status"] = air_status
            result["air_color"] = f"{{{self._color_to_code(air_color)}}}"
        
        if owm_data:
            vis_mi = round(owm_data["visibility_m"] / 1609.34, 1)
            result["visibility"] = f"{vis_mi}mi"
            
            is_foggy, fog_status, fog_color = self.determine_fog_status(
                owm_data["visibility_m"],
                owm_data["humidity"],
                owm_data["temperature_f"]
            )
            result["is_foggy"] = "Yes" if is_foggy else "No"
            result["fog_status"] = fog_status
            result["fog_color"] = f"{{{self._color_to_code(fog_color)}}}"
        
        if pollen_data:
            result["grass_pollen"] = pollen_data["grass_pollen"]
            result["grass_pollen_level"] = pollen_data["grass_pollen_level"]
            result["grass_pollen_color"] = f"{{{self._color_to_code(pollen_data['grass_pollen_color'])}}}"
            result["tree_pollen"] = pollen_data["tree_pollen"]
            result["tree_pollen_level"] = pollen_data["tree_pollen_level"]
            result["tree_pollen_color"] = f"{{{self._color_to_code(pollen_data['tree_pollen_color'])}}}"
            result["weed_pollen"] = pollen_data["weed_pollen"]
            result["weed_pollen_level"] = pollen_data["weed_pollen_level"]
            result["weed_pollen_color"] = f"{{{self._color_to_code(pollen_data['weed_pollen_color'])}}}"
        
        # Build formatted message
        parts = []
        if result["aqi"] is not None:
            parts.append(f"AQI:{result['aqi']}")
        if result["visibility"]:
            parts.append(f"VIS:{result['visibility']}")
        if result["grass_pollen"] is not None:
            parts.append(f"GRASS:{result['grass_pollen']}")
        if result["tree_pollen"] is not None:
            parts.append(f"TREES:{result['tree_pollen']}")
        if result["weed_pollen"] is not None:
            parts.append(f"WEEDS:{result['weed_pollen']}")
        result["formatted"] = " ".join(parts) if parts else "NO DATA"
        
        return PluginResult(
            available=True,
            data=result
        )
    
    def _color_to_code(self, color: str) -> int:
        """Convert color name to board code."""
        color_map = {
            "GREEN": 66,
            "YELLOW": 65,
            "ORANGE": 64,
            "RED": 63,
            "PURPLE": 68,
            "MAROON": 68,
        }
        return color_map.get(color.upper(), 66)


# Export the plugin class
Plugin = AirFogPlugin

