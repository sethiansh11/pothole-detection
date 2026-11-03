import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import Point


# ==================================================
# CONFIGURATION
# ==================================================

# Defaults are used only when you press Enter at the coordinate prompts.
# You can enter any other valid latitude/longitude when running the script.
DEFAULT_LATITUDE = 30.35360
DEFAULT_LONGITUDE = 76.36062

ROAD_SEARCH_RADIUS = 1000       # metres
FEATURE_SEARCH_RADIUS = 1000    # metres
NETWORK_TYPE = "all"

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

# Manual OSM category corrections confirmed during development.
# This applies only to this specific OSM feature, wherever it is retrieved.
MANUAL_CATEGORY_OVERRIDES = {
    396666410: "sports_ground",
}


# ==================================================
# OPENSTREETMAP FEATURE TAGS
# ==================================================

FEATURE_TAGS = {
    "amenity": [
        "hospital",
        "school",
        "college",
        "university",
        "kindergarten",
        "police",
        "fire_station",
        "bus_station",
    ],
    "healthcare": ["hospital"],
    "highway": ["bus_stop"],
    "public_transport": ["platform", "station"],
    "emergency": ["fire_station"],
    "railway": ["station"],
    "leisure": [
        "pitch",
        "sports_centre",
        "stadium",
        "track",
        "recreation_ground",
    ],
    "landuse": ["recreation_ground"],
}

FEATURE_CATEGORIES = [
    "hospital",
    "school",
    "kindergarten",
    "college",
    "university",
    "sports_ground",
    "police_station",
    "fire_station",
    "bus_stop",
    "bus_station",
    "other_public_transport",
]


# ==================================================
# GEOGRAPHIC SEVERITY V1 CONFIGURATION
# ==================================================

# Initial heuristic weights. These require validation/calibration against
# multiple pothole locations before being treated as reliable.
GEO_WEIGHTS = {
    "road_importance": 0.40,
    "critical_infrastructure": 0.35,
    "public_activity": 0.25,
}

ROAD_IMPORTANCE_SCORES = {
    "motorway": 100,
    "trunk": 95,
    "primary": 90,
    "secondary": 80,
    "tertiary": 70,
    "unclassified": 50,
    "residential": 45,
    "living_street": 35,
    "service": 30,
    "track": 20,
    "footway": 10,
    "path": 10,
}

# A manually verified internal road can be supplied for a specific run.
MANUAL_ROAD_USE_SCORE = {
    "internal_vehicle_access": 30,
}

CRITICAL_INFRASTRUCTURE_CATEGORIES = [
    "hospital",
    "police_station",
    "fire_station",
]

PUBLIC_ACTIVITY_CATEGORIES = [
    "school",
    "kindergarten",
    "college",
    "university",
    "sports_ground",
    "bus_stop",
    "bus_station",
    "other_public_transport",
]


# ==================================================
# 1. INPUT AND VALIDATION
# ==================================================

def validate_coordinates(latitude, longitude):
    """Raise ValueError if coordinates are outside valid geographic ranges."""
    if not -90 <= latitude <= 90:
        raise ValueError(
            f"Latitude must be between -90 and 90. Received: {latitude}"
        )

    if not -180 <= longitude <= 180:
        raise ValueError(
            f"Longitude must be between -180 and 180. Received: {longitude}"
        )


def prompt_for_coordinate(label, default_value, minimum, maximum):
    """Prompt until a valid coordinate is entered; Enter selects the default."""
    while True:
        raw_value = input(
            f"Enter {label} [{default_value}]: "
        ).strip()

        if not raw_value:
            return float(default_value)

        try:
            value = float(raw_value)
        except ValueError:
            print("Please enter a valid number.")
            continue

        if not minimum <= value <= maximum:
            print(
                f"{label.capitalize()} must be between {minimum} and {maximum}."
            )
            continue

        return value


def prompt_for_run_inputs():
    """Collect coordinates and optional location-specific context."""
    print("\n========== NEW POTHOLE LOCATION ==========")
    print("Press Enter to use the previous Thapar test coordinates as defaults.")

    latitude = prompt_for_coordinate(
        "latitude", DEFAULT_LATITUDE, -90, 90
    )
    longitude = prompt_for_coordinate(
        "longitude", DEFAULT_LONGITUDE, -180, 180
    )

    print("\nOptional road-use verification:")
    print(
        "Press Enter to use the OpenStreetMap road classification. "
        "For the verified Thapar internal road, enter: internal_vehicle_access"
    )
    observed_road_use = input(
        "Observed road use (optional): "
    ).strip().lower() or None

    valid_manual_road_uses = (
        set(ROAD_IMPORTANCE_SCORES)
        | set(MANUAL_ROAD_USE_SCORE)
    )

    if observed_road_use and observed_road_use not in valid_manual_road_uses:
        print(
            f"Unrecognized road-use value '{observed_road_use}'. "
            "The engine will use the OpenStreetMap classification instead."
        )
        observed_road_use = None

    site_context = input(
        "Site/context name (optional, e.g. Thapar campus): "
    ).strip() or None

    validate_coordinates(latitude, longitude)

    return latitude, longitude, observed_road_use, site_context


# ==================================================
# 2. GENERAL HELPERS
# ==================================================

def get_tag_values(value):
    """Normalize a single OSM tag or a list of OSM tag values into a set."""
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]

    result = set()
    for item in values:
        if item is None:
            continue
        try:
            if pd.isna(item):
                continue
        except (TypeError, ValueError):
            pass
        result.add(str(item).strip().lower())

    return result


def clean_value(value):
    """Convert pandas/NumPy missing values and collections to CSV-friendly values."""
    if value is None:
        return None

    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def get_osm_id(row):
    """Get the OSM feature ID after OSMnx's reset_index()."""
    value = row.get("id", row.get("osmid"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ==================================================
# 3. OVERPASS REQUEST WITH RETRIES
# ==================================================

def run_with_retries(request_function, description, max_retries=4):
    """Retry temporary Overpass/API errors using exponential backoff."""
    for attempt in range(max_retries):
        try:
            print(f"\nDownloading {description}...")
            result = request_function()
            print(f"{description} downloaded successfully.")
            return result

        except Exception as exc:
            error = str(exc)
            temporary_error = any(
                code in error for code in ["429", "500", "502", "503", "504"]
            )

            if not temporary_error or attempt == max_retries - 1:
                raise

            wait_seconds = 5 * (2 ** attempt)
            print(
                f"Overpass server unavailable. Retrying in {wait_seconds} seconds..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"Unable to retrieve {description}.")


# ==================================================
# 4. ROAD NETWORK RETRIEVAL AND MATCHING
# ==================================================

def get_road_graph(latitude, longitude):
    center = (latitude, longitude)

    return run_with_retries(
        lambda: ox.graph_from_point(
            center,
            dist=ROAD_SEARCH_RADIUS,
            network_type=NETWORK_TYPE,
            retain_all=True,
        ),
        "road network",
    )


def find_nearest_mapped_way(latitude, longitude, projected_graph):
    point = gpd.GeoSeries(
        [Point(longitude, latitude)],
        crs="EPSG:4326",
    ).to_crs(projected_graph.graph["crs"]).iloc[0]

    edges = ox.graph_to_gdfs(
        projected_graph,
        nodes=False,
        edges=True,
    ).reset_index()

    edges["distance_m"] = edges.geometry.distance(point)

    # Remove reverse-direction duplicates from the diagnostic display.
    edges["physical_id"] = edges.apply(
        lambda row: (
            str(row["osmid"]),
            tuple(sorted((str(row["u"]), str(row["v"])))),
        ),
        axis=1,
    )

    nearby = (
        edges.sort_values("distance_m")
        .drop_duplicates("physical_id")
        .head(5)
    )

    print("\n--- NEAREST MAPPED WAYS ---")
    for _, way in nearby.iterrows():
        print(
            {
                "name": clean_value(way.get("name")),
                "osmid": clean_value(way.get("osmid")),
                "type": clean_value(way.get("highway")),
                "distance_m": round(float(way["distance_m"]), 2),
            }
        )

    edge, distance = ox.distance.nearest_edges(
        projected_graph,
        X=point.x,
        Y=point.y,
        return_dist=True,
    )
    u, v, key = edge
    way = projected_graph.edges[u, v, key]

    return {
        "osm_id": way.get("osmid"),
        "name": clean_value(way.get("name")),
        "highway_type": way.get("highway"),
        "surface": clean_value(way.get("surface")),
        "distance_m": round(float(distance), 2),
    }


# ==================================================
# 5. INFRASTRUCTURE CLASSIFICATION
# ==================================================

def classify_osm_feature(row):
    """Classify a feature using its original OpenStreetMap tags."""
    leisure = get_tag_values(row.get("leisure"))
    landuse = get_tag_values(row.get("landuse"))
    amenity = get_tag_values(row.get("amenity"))
    healthcare = get_tag_values(row.get("healthcare"))
    highway = get_tag_values(row.get("highway"))
    public_transport = get_tag_values(row.get("public_transport"))
    emergency = get_tag_values(row.get("emergency"))
    railway = get_tag_values(row.get("railway"))

    if "hospital" in amenity or "hospital" in healthcare:
        return "hospital"

    if leisure.intersection(
        {"pitch", "sports_centre", "stadium", "track", "recreation_ground"}
    ) or "recreation_ground" in landuse:
        return "sports_ground"

    if "school" in amenity:
        return "school"
    if "kindergarten" in amenity:
        return "kindergarten"
    if "college" in amenity:
        return "college"
    if "university" in amenity:
        return "university"

    if "police" in amenity:
        return "police_station"

    if "fire_station" in amenity or "fire_station" in emergency:
        return "fire_station"

    if "bus_stop" in highway:
        return "bus_stop"

    if "bus_station" in amenity:
        return "bus_station"

    if public_transport.intersection({"platform", "station"}) or "station" in railway:
        return "other_public_transport"

    return None


def classify_feature_with_overrides(row):
    """Apply user-verified category overrides while retaining the OSM category."""
    osm_id = get_osm_id(row)
    if osm_id in MANUAL_CATEGORY_OVERRIDES:
        return MANUAL_CATEGORY_OVERRIDES[osm_id]
    return classify_osm_feature(row)


def get_classification_source(row):
    osm_id = get_osm_id(row)
    if osm_id in MANUAL_CATEGORY_OVERRIDES:
        return "user_verified_manual_override"
    return "openstreetmap_tags"


# ==================================================
# 6. SPATIAL CONTAINMENT
# ==================================================

def is_inside_facility(geometry, point):
    if geometry is None or geometry.is_empty:
        return False

    if geometry.geom_type not in ("Polygon", "MultiPolygon"):
        return False

    return bool(geometry.covers(point))


def get_boundary_distance(geometry, point):
    if geometry is None or geometry.is_empty:
        return None

    if geometry.geom_type not in ("Polygon", "MultiPolygon"):
        return None

    return round(float(geometry.boundary.distance(point)), 2)


# ==================================================
# 7. RETRIEVE NEARBY INFRASTRUCTURE
# ==================================================

def empty_infrastructure_summary():
    return {
        category: {"count": 0, "inside_count": 0, "nearest": None}
        for category in FEATURE_CATEGORIES
    }


def get_nearby_infrastructure(latitude, longitude):
    center = (latitude, longitude)

    features = run_with_retries(
        lambda: ox.features.features_from_point(
            center_point=center,
            tags=FEATURE_TAGS,
            dist=FEATURE_SEARCH_RADIUS,
        ),
        "nearby infrastructure",
    )

    if features.empty:
        print("\nNo infrastructure features returned by OpenStreetMap.")
        return empty_infrastructure_summary(), []

    features = features.reset_index()

    # Keep both the original OSM classification and the verified category.
    features["osm_category"] = features.apply(classify_osm_feature, axis=1)
    features["category"] = features.apply(classify_feature_with_overrides, axis=1)
    features["classification_source"] = features.apply(
        get_classification_source,
        axis=1,
    )

    features = features[features["category"].notna()].copy()

    valid_geometry = features.geometry.notna() & ~features.geometry.is_empty
    features = features[valid_geometry].copy()

    if features.empty:
        return empty_infrastructure_summary(), []

    # Project into metres before calculating distances.
    projected_features = ox.projection.project_gdf(features)

    projected_point = gpd.GeoSeries(
        [Point(longitude, latitude)],
        crs="EPSG:4326",
    ).to_crs(projected_features.crs).iloc[0]

    projected_features["distance_m"] = projected_features.geometry.distance(
        projected_point
    )
    projected_features["inside_facility"] = projected_features.geometry.apply(
        lambda geometry: is_inside_facility(geometry, projected_point)
    )
    projected_features["boundary_distance_m"] = projected_features.geometry.apply(
        lambda geometry: get_boundary_distance(geometry, projected_point)
    )

    projected_features = projected_features[
        projected_features["distance_m"] <= FEATURE_SEARCH_RADIUS
    ].copy()

    summary = {}
    for category in FEATURE_CATEGORIES:
        category_features = (
            projected_features[projected_features["category"] == category]
            .sort_values("distance_m")
        )

        count = len(category_features)
        inside_count = int(category_features["inside_facility"].sum())

        if count == 0:
            summary[category] = {
                "count": 0,
                "inside_count": 0,
                "nearest": None,
            }
            continue

        nearest = category_features.iloc[0]
        summary[category] = {
            "count": count,
            "inside_count": inside_count,
            "nearest": {
                "name": clean_value(nearest.get("name")),
                "osm_id": clean_value(nearest.get("id", nearest.get("osmid"))),
                "osm_element": clean_value(nearest.get("element")),
                "distance_m": round(float(nearest["distance_m"]), 2),
                "inside_facility": bool(nearest["inside_facility"]),
                "boundary_distance_m": clean_value(
                    nearest["boundary_distance_m"]
                ),
            },
        }

    facility_records = []
    for _, feature in projected_features.iterrows():
        facility_records.append(
            {
                "category": feature["category"],
                "osm_category": feature["osm_category"],
                "classification_source": feature["classification_source"],
                "name": clean_value(feature.get("name")),
                "osm_id": clean_value(feature.get("id", feature.get("osmid"))),
                "osm_element": clean_value(feature.get("element")),
                "osm_amenity": clean_value(feature.get("amenity")),
                "osm_healthcare": clean_value(feature.get("healthcare")),
                "osm_highway": clean_value(feature.get("highway")),
                "osm_leisure": clean_value(feature.get("leisure")),
                "osm_landuse": clean_value(feature.get("landuse")),
                "osm_sport": clean_value(feature.get("sport")),
                "distance_m": round(float(feature["distance_m"]), 2),
                "inside_facility": bool(feature["inside_facility"]),
                "boundary_distance_m": clean_value(
                    feature["boundary_distance_m"]
                ),
            }
        )

    return summary, facility_records


# ==================================================
# 8. GEOGRAPHIC SEVERITY SCORING — V1
# ==================================================

def calculate_road_importance(road_result, observed_road_use=None):
    """Use a per-run manual road-use value if supplied; otherwise use OSM tags."""
    if observed_road_use:
        manual_value = observed_road_use.strip().lower()

        if manual_value in MANUAL_ROAD_USE_SCORE:
            return (
                float(MANUAL_ROAD_USE_SCORE[manual_value]),
                f"User-verified road use: {manual_value}",
            )

        if manual_value in ROAD_IMPORTANCE_SCORES:
            return (
                float(ROAD_IMPORTANCE_SCORES[manual_value]),
                f"User-verified road class: {manual_value}",
            )

    road_types = get_tag_values(road_result.get("highway_type"))
    matching_scores = [
        ROAD_IMPORTANCE_SCORES[tag]
        for tag in road_types
        if tag in ROAD_IMPORTANCE_SCORES
    ]

    if not matching_scores:
        return None, "Road importance unavailable: no recognized road class"

    selected_score = max(matching_scores)
    return (
        float(selected_score),
        f"OSM road class: {', '.join(sorted(road_types))}",
    )


def calculate_facility_proximity(nearest, category):
    """Convert distance to a provisional proximity-impact score."""
    if nearest is None:
        return None

    if bool(nearest.get("inside_facility", False)):
        return 100.0

    distance = nearest.get("distance_m")
    if distance is None or pd.isna(distance):
        return None

    distance = float(distance)

    if category in CRITICAL_INFRASTRUCTURE_CATEGORIES:
        if distance <= 100:
            return 100.0
        if distance <= 250:
            return 80.0
        if distance <= 500:
            return 60.0
        if distance <= 1000:
            return 30.0
        return 0.0

    if distance <= 50:
        return 100.0
    if distance <= 100:
        return 90.0
    if distance <= 250:
        return 75.0
    if distance <= 500:
        return 50.0
    if distance <= 1000:
        return 25.0
    return 0.0


def calculate_infrastructure_group_score(infrastructure_summary, categories):
    """Use the strongest observed proximity score in a feature group."""
    candidates = []

    for category in categories:
        category_data = infrastructure_summary.get(category, {})
        nearest = category_data.get("nearest")
        if nearest is None:
            continue

        score = calculate_facility_proximity(nearest, category)
        if score is None:
            continue

        candidates.append(
            {
                "category": category,
                "score": score,
                "distance_m": nearest.get("distance_m"),
                "name": nearest.get("name"),
                "inside_facility": nearest.get("inside_facility", False),
            }
        )

    if not candidates:
        return None, "No matching mapped features available"

    best = max(candidates, key=lambda item: item["score"])
    description = (
        f"{best['category']} - {best['name'] or 'unnamed'} "
        f"({best['distance_m']} m)"
    )
    return float(best["score"]), description


def calculate_geo_severity(
    road_result,
    infrastructure_summary,
    observed_road_use=None,
):
    """Calculate the provisional 0-100 geographic impact score."""
    road_score, road_basis = calculate_road_importance(
        road_result,
        observed_road_use,
    )

    critical_score, critical_basis = calculate_infrastructure_group_score(
        infrastructure_summary,
        CRITICAL_INFRASTRUCTURE_CATEGORIES,
    )

    activity_score, activity_basis = calculate_infrastructure_group_score(
        infrastructure_summary,
        PUBLIC_ACTIVITY_CATEGORIES,
    )

    components = {
        "road_importance": road_score,
        "critical_infrastructure": critical_score,
        "public_activity": activity_score,
    }

    available = {
        key: value
        for key, value in components.items()
        if value is not None
    }

    if not available:
        raise ValueError(
            "Cannot calculate geographic severity: no scoring dimensions have data."
        )

    # Renormalize weights only when an entire scoring dimension is unavailable.
    available_weight = sum(GEO_WEIGHTS[key] for key in available)
    weighted_score = sum(
        value * GEO_WEIGHTS[key]
        for key, value in available.items()
    ) / available_weight
    geo_score = round(weighted_score, 2)

    if geo_score < 35:
        impact_level = "LOW"
    elif geo_score < 60:
        impact_level = "MODERATE"
    elif geo_score < 80:
        impact_level = "HIGH"
    else:
        impact_level = "VERY_HIGH"

    missing_critical_categories = [
        category
        for category in CRITICAL_INFRASTRUCTURE_CATEGORIES
        if infrastructure_summary.get(category, {}).get("nearest") is None
    ]

    missing_activity_categories = [
        category
        for category in PUBLIC_ACTIVITY_CATEGORIES
        if infrastructure_summary.get(category, {}).get("nearest") is None
    ]

    return {
        "road_importance_score": road_score,
        "critical_infrastructure_score": critical_score,
        "public_activity_score": activity_score,
        "geo_severity_score": geo_score,
        "geo_severity_level": impact_level,
        "road_score_basis": road_basis,
        "critical_score_basis": critical_basis,
        "activity_score_basis": activity_basis,
        "scoring_method": "heuristic_rule_based_v1",
        "scoring_dimensions_available": ", ".join(available.keys()),
        "critical_categories_without_osm_match": ", ".join(
            missing_critical_categories
        ),
        "activity_categories_without_osm_match": ", ".join(
            missing_activity_categories
        ),
        "data_quality": "PROVISIONAL_OSM_DATA",
        "data_quality_note": (
            "Initial heuristic scores; OSM coverage may be incomplete. "
            "No live traffic counts are used. Missing OSM features do not "
            "prove that real-world facilities are absent."
        ),
    }


# ==================================================
# 9. CSV OUTPUT HELPERS
# ==================================================

def append_records_to_csv(path, records, columns=None):
    """Append records while preserving previous pothole runs."""
    new_data = pd.DataFrame(records, columns=columns)

    if path.exists():
        try:
            existing_data = pd.read_csv(path)
            combined = pd.concat(
                [existing_data, new_data],
                ignore_index=True,
                sort=False,
            )
        except pd.errors.EmptyDataError:
            combined = new_data
    else:
        combined = new_data

    combined.to_csv(path, index=False)


def save_results(
    road_result,
    infrastructure_summary,
    facility_records,
    latitude,
    longitude,
    observed_road_use,
    site_context,
    geo_severity_result,
    report_id,
    captured_at_utc,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_row = {
        "report_id": report_id,
        "captured_at_utc": captured_at_utc,
        "latitude": latitude,
        "longitude": longitude,
        "observed_road_use": observed_road_use,
        "verified_site_context": site_context,
        "site_context_source": "user_entered" if site_context else "not_provided",
        "nearest_way_osm_id": clean_value(road_result["osm_id"]),
        "osm_highway_type": clean_value(road_result["highway_type"]),
        "nearest_way_distance_m": road_result["distance_m"],
        "feature_search_radius_m": FEATURE_SEARCH_RADIUS,
    }

    for category, data in infrastructure_summary.items():
        summary_row[f"{category}_count"] = data["count"]
        summary_row[f"{category}_inside_count"] = data["inside_count"]

        nearest = data["nearest"]
        if nearest is None:
            summary_row[f"nearest_{category}_name"] = None
            summary_row[f"nearest_{category}_distance_m"] = None
            summary_row[f"nearest_{category}_osm_id"] = None
            summary_row[f"nearest_{category}_inside"] = None
            summary_row[f"nearest_{category}_boundary_distance_m"] = None
        else:
            summary_row[f"nearest_{category}_name"] = nearest["name"]
            summary_row[f"nearest_{category}_distance_m"] = nearest["distance_m"]
            summary_row[f"nearest_{category}_osm_id"] = nearest["osm_id"]
            summary_row[f"nearest_{category}_inside"] = nearest["inside_facility"]
            summary_row[f"nearest_{category}_boundary_distance_m"] = (
                nearest["boundary_distance_m"]
            )

    summary_row["total_transport_facilities"] = sum(
        infrastructure_summary[category]["count"]
        for category in [
            "bus_stop",
            "bus_station",
            "other_public_transport",
        ]
    )

    summary_row.update(geo_severity_result)

    summary_path = OUTPUT_DIR / "geo_features.csv"
    append_records_to_csv(summary_path, [summary_row])

    # Link every nearby-facility row to the pothole report that produced it.
    detailed_records = []
    for record in facility_records:
        detailed_record = {
            "report_id": report_id,
            "captured_at_utc": captured_at_utc,
            "latitude": latitude,
            "longitude": longitude,
        }
        detailed_record.update(record)
        detailed_records.append(detailed_record)

    details_path = OUTPUT_DIR / "geo_nearby_facilities.csv"
    facility_columns = [
        "report_id",
        "captured_at_utc",
        "latitude",
        "longitude",
        "category",
        "osm_category",
        "classification_source",
        "name",
        "osm_id",
        "osm_element",
        "osm_amenity",
        "osm_healthcare",
        "osm_highway",
        "osm_leisure",
        "osm_landuse",
        "osm_sport",
        "distance_m",
        "inside_facility",
        "boundary_distance_m",
    ]
    append_records_to_csv(
        details_path,
        detailed_records,
        columns=facility_columns,
    )

    print("\n--- FILES UPDATED (PREVIOUS REPORTS PRESERVED) ---")
    print(summary_path)
    print(details_path)


# ==================================================
# 10. RUN THE ENGINE FOR ANY COORDINATES
# ==================================================

def run_geo_severity(
    latitude,
    longitude,
    observed_road_use=None,
    site_context=None,
):
    """Run the complete Geo-Severity pipeline for one pothole location.

    This function can be called by the future application or another module:
        run_geo_severity(30.35360, 76.36062,
                         observed_road_use="internal_vehicle_access",
                         site_context="Thapar campus")
    """
    latitude = float(latitude)
    longitude = float(longitude)
    validate_coordinates(latitude, longitude)

    report_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    captured_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print("\n========== GEO LOCATION ENGINE ==========")
    print(f"Report ID: {report_id}")
    print(f"Latitude: {latitude}")
    print(f"Longitude: {longitude}")
    print(f"Site context: {site_context or 'Not provided'}")
    print(f"Manual road-use override: {observed_road_use or 'Using OSM classification'}")

    # 1. Retrieve the road network and find the nearest mapped way.
    graph = get_road_graph(latitude, longitude)
    projected_graph = ox.project_graph(graph)

    road_result = find_nearest_mapped_way(
        latitude,
        longitude,
        projected_graph,
    )

    print("\n--- ROAD INFORMATION ---")
    for key, value in road_result.items():
        print(f"{key}: {value}")

    # 2. Retrieve nearby geographic infrastructure.
    infrastructure_summary, facility_records = get_nearby_infrastructure(
        latitude,
        longitude,
    )

    print("\n--- NEARBY INFRASTRUCTURE ---")
    for category, data in infrastructure_summary.items():
        print(f"\n{category.upper()}")
        print(f"Count: {data['count']}")
        print(f"Inside facility count: {data['inside_count']}")

        nearest = data["nearest"]
        if nearest is None:
            print("No mapped feature found within radius.")
            continue

        print(f"Nearest: {nearest['name'] or 'Unnamed'}")
        print(f"Distance: {nearest['distance_m']} metres")
        print(f"Inside facility: {nearest['inside_facility']}")
        print(f"Boundary distance: {nearest['boundary_distance_m']}")

    # 3. Calculate geographic severity.
    geo_severity_result = calculate_geo_severity(
        road_result,
        infrastructure_summary,
        observed_road_use=observed_road_use,
    )

    print("\n========== GEO-SEVERITY RESULTS ==========")
    for key, value in geo_severity_result.items():
        print(f"{key}: {value}")

    # 4. Save this report without overwriting previous locations.
    save_results(
        road_result=road_result,
        infrastructure_summary=infrastructure_summary,
        facility_records=facility_records,
        latitude=latitude,
        longitude=longitude,
        observed_road_use=observed_road_use,
        site_context=site_context,
        geo_severity_result=geo_severity_result,
        report_id=report_id,
        captured_at_utc=captured_at_utc,
    )

    print("\nGeo-location extraction and scoring completed.")

    return {
        "report_id": report_id,
        "latitude": latitude,
        "longitude": longitude,
        "road_result": road_result,
        "infrastructure_summary": infrastructure_summary,
        "geo_severity": geo_severity_result,
    }


# ==================================================
# 11. MAIN EXECUTION
# ==================================================

def main():
    latitude, longitude, observed_road_use, site_context = (
        prompt_for_run_inputs()
    )

    run_geo_severity(
        latitude=latitude,
        longitude=longitude,
        observed_road_use=observed_road_use,
        site_context=site_context,
    )


if __name__ == "__main__":
    main()
