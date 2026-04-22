"""Weather tool — Open-Meteo (free, no API key).

Open-Meteo's forecast API only supports up to ~16 days into the future. For trips
further out we fall back to the historical archive for the same dates one year
prior, which gives a realistic seasonal proxy ("what was the weather like in
Paris in late April last year").

The `data_source` field on the result tells the caller (and the LLM, and the UI)
exactly which path was taken:
  - "live_forecast"                          → in-horizon, forecast call succeeded
  - "historical_proxy_last_year"             → trip is beyond 16 days, intentional proxy
  - "historical_fallback_after_forecast_error" → in-horizon but Open-Meteo's
                                                 forecast endpoint failed (e.g. 504),
                                                 served same-dates-last-year as a fallback
"""
import time
import requests
from datetime import datetime, date as date_type, timedelta

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_HORIZON_DAYS = 16
FORECAST_RETRIES = 3
FORECAST_RETRY_DELAY = 1.5  # seconds, doubled each retry


def get_weather(latitude: float, longitude: float, start_date: str, end_date: str) -> dict:
    """Fetch weather for a date range.

    - In-horizon dates → live forecast (with retries on transient errors).
    - Beyond horizon → same-dates-last-year as a seasonal proxy.
    - In-horizon but the forecast API is having an outage → fall back to the
      historical proxy and label it as such, instead of erroring out the agent.
    """
    today = date_type.today()
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError as e:
        return {"error": f"Invalid date format: {e}"}

    horizon_cutoff = today + timedelta(days=FORECAST_HORIZON_DAYS)

    if start <= horizon_cutoff and end <= horizon_cutoff and end >= today:
        try:
            return _fetch_forecast(latitude, longitude, start_date, end_date)
        except Exception as forecast_err:
            # Forecast API outage — gracefully fall back to historical proxy and
            # label it so the UI banner shows the right message.
            try:
                fallback = _fetch_historical_proxy(latitude, longitude, start, end)
                fallback["data_source"] = "historical_fallback_after_forecast_error"
                fallback["note"] = (
                    f"Open-Meteo's live forecast endpoint was unavailable "
                    f"({forecast_err}). The figures below are the actual weather "
                    f"recorded on the same calendar dates one year ago — treat them "
                    f"as a seasonal estimate and recheck closer to your trip once "
                    f"the live forecast service recovers."
                )
                return fallback
            except Exception as hist_err:
                return {
                    "error": f"Forecast failed ({forecast_err}) and historical "
                             f"fallback also failed ({hist_err})."
                }

    # Beyond horizon (or past) — historical proxy is the intended source
    return _fetch_historical_proxy(latitude, longitude, start, end)


def _fetch_forecast(latitude: float, longitude: float, start_date: str, end_date: str) -> dict:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode",
        "temperature_unit": "fahrenheit",
        "timezone": "auto",
    }
    # Retry on transient 5xx and connection errors
    delay = FORECAST_RETRY_DELAY
    last_exc = None
    for attempt in range(FORECAST_RETRIES):
        try:
            resp = requests.get(FORECAST_URL, params=params, timeout=10)
            if resp.status_code >= 500:
                raise requests.HTTPError(f"{resp.status_code} from Open-Meteo")
            resp.raise_for_status()
            data = resp.json()
            break
        except (requests.RequestException, requests.HTTPError) as e:
            last_exc = e
            if attempt < FORECAST_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise last_exc
    daily = data.get("daily", {})

    forecasts = []
    for i, d in enumerate(daily.get("time", [])):
        forecasts.append({
            "date": d,
            "temp_high_f": daily["temperature_2m_max"][i],
            "temp_low_f": daily["temperature_2m_min"][i],
            "precipitation_chance": daily["precipitation_probability_max"][i],
            "condition": _weather_code_to_text(daily["weathercode"][i]),
        })

    return {
        "location": f"{latitude},{longitude}",
        "data_source": "live_forecast",
        "forecasts": forecasts,
    }


def _fetch_historical_proxy(latitude: float, longitude: float,
                            orig_start: date_type, orig_end: date_type) -> dict:
    """Pull last year's data for the same calendar dates as a seasonal proxy."""
    try:
        hist_start = orig_start.replace(year=orig_start.year - 1)
        hist_end = orig_end.replace(year=orig_end.year - 1)
    except ValueError:
        # Leap-day edge case
        hist_start = orig_start.replace(year=orig_start.year - 1, day=28)
        hist_end = orig_end.replace(year=orig_end.year - 1, day=28)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": hist_start.isoformat(),
        "end_date": hist_end.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
        "temperature_unit": "fahrenheit",
        "timezone": "auto",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily", {})

    forecasts = []
    hist_dates = daily.get("time", [])
    trip_days = (orig_end - orig_start).days + 1

    for i in range(min(len(hist_dates), trip_days)):
        precip_mm = daily["precipitation_sum"][i] or 0
        # Crude probability estimate from precipitation amount
        if precip_mm == 0:
            chance = 5
        elif precip_mm < 1:
            chance = 25
        elif precip_mm < 5:
            chance = 60
        else:
            chance = 90

        trip_date = (orig_start + timedelta(days=i)).isoformat()
        forecasts.append({
            "date": trip_date,
            "temp_high_f": daily["temperature_2m_max"][i],
            "temp_low_f": daily["temperature_2m_min"][i],
            "precipitation_chance": chance,
            "precipitation_sum_mm_last_year": precip_mm,
            "condition": _weather_code_to_text(daily["weathercode"][i]),
        })

    return {
        "location": f"{latitude},{longitude}",
        "data_source": "historical_proxy_last_year",
        "note": (
            f"Trip dates ({orig_start} to {orig_end}) are beyond Open-Meteo's "
            f"16-day forecast horizon. Showing actual weather from the SAME dates "
            f"one year ago ({hist_start} to {hist_end}) as a seasonal proxy. "
            f"Use this to plan typical conditions, but check a real forecast closer to the trip."
        ),
        "forecasts": forecasts,
    }


def _weather_code_to_text(code: int) -> str:
    """Convert WMO weather code to human-readable text."""
    mapping = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Foggy", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
    }
    return mapping.get(code, f"Unknown ({code})")
