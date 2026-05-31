"""Catalog builders for pollutants and land-cover subclasses."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..settings import SensorAnalysisSettings


METAL_KEYWORDS = (
    "aluminum",
    "arsenic",
    "barium",
    "beryllium",
    "boron",
    "cadmium",
    "chromium",
    "cobalt",
    "copper",
    "iron",
    "lead",
    "lithium",
    "manganese",
    "mercury",
    "nickel",
    "selenium",
    "silver",
    "tin",
    "uranium",
    "vanadium",
    "zinc",
    "bismuth",
)

ORGANIC_KEYWORDS = (
    "benz",
    "chlor",
    "phenol",
    "dichloro",
    "trichloro",
    "tetrachloro",
    "carbon_tetrachloride",
    "aldrin",
    "chlordane",
    "ddt",
    "dieldrin",
    "endrin",
    "endosulfan",
    "heptachlor",
    "lindane",
    "methoxychlor",
    "toxaphene",
    "demeton",
    "guthion",
    "malathion",
    "parathion",
    "carbaryl",
    "methyl_parathion",
    "ethion",
    "disulfoton",
    "phosdrin",
    "diazinon",
    "azinphos",
    "bhc",
    "2_4_",
    "pentachlorophenol",
    "polychlorinated_biphenyls",
    "organochlorine",
    "organophosphorus",
)

MICROBIOLOGICAL_KEYWORDS = (
    "coliform",
    "bacteria",
    "streptococci",
    "salmonella",
    "coliphages",
    "protozoa",
    "fungi",
    "algae",
    "phytoplankton",
    "zooplankton",
    "cyanobacteria",
    "escherichia",
)

NUTRIENT_KEYWORDS = (
    "nitrogen",
    "ammonia",
    "ammoniacal",
    "nitrate",
    "nitrite",
    "phosph",
    "phosphate",
    "kjeldahl",
    "albuminoid",
)

OXYGEN_ORGANIC_KEYWORDS = (
    "oxygen_demand",
    "organic_carbon",
    "hydrocarbons",
    "detergents",
    "oils_and_grease",
)

PHYSICOCHEMICAL_KEYWORDS = (
    "ph",
    "turbidity",
    "conductivity",
    "hardness",
    "solids",
    "alkalinity",
    "chlorides",
    "sulfates",
    "sulfides",
    "fluorides",
    "acidity",
    "transparency",
    "oxygen_saturation",
    "color",
)


@dataclass(frozen=True)
class PollutantDefinition:
    """Metadata for one modeled pollutant."""

    name: str
    transform: str
    expression: str
    type_group: str
    observation_count: int
    importance_group: str


@dataclass(frozen=True)
class LandCoverSubclassDefinition:
    """Metadata for one land-cover subclass."""

    identifier: str
    label: str


def _classify_pollutant_type(name: str) -> str:
    lowered = name.lower()
    if lowered == "water_quality_index":
        return "composite_indices"
    if any(keyword in lowered for keyword in NUTRIENT_KEYWORDS):
        return "nutrients"
    if any(keyword in lowered for keyword in OXYGEN_ORGANIC_KEYWORDS):
        return "oxygen_demand_organic_load"
    if any(keyword in lowered for keyword in MICROBIOLOGICAL_KEYWORDS):
        return "microbiological"
    if any(keyword in lowered for keyword in METAL_KEYWORDS):
        return "metals"
    if any(keyword in lowered for keyword in ORGANIC_KEYWORDS):
        return "organics_pesticides"
    if any(keyword in lowered for keyword in PHYSICOCHEMICAL_KEYWORDS):
        return "core_physicochemical"
    return "other"


def _importance_group(observation_count: int, settings: SensorAnalysisSettings) -> str:
    for tier in settings.importance_tiers:
        if observation_count >= tier.minimum_observations:
            return tier.name
    return settings.importance_tiers[-1].name


def build_pollutant_catalog(
    sensor_data: pd.DataFrame,
    transformations: dict[str, dict[str, object]],
    settings: SensorAnalysisSettings,
) -> list[PollutantDefinition]:
    """Build the pollutant catalog from the sensor schema and transform metadata."""
    pollutants: list[PollutantDefinition] = []
    excluded = set(settings.excluded_pollutant_columns)
    for name, recommendation in sorted(transformations.items()):
        if name in excluded or name not in sensor_data.columns:
            continue
        observation_count = int(sensor_data[name].notna().sum())
        pollutants.append(
            PollutantDefinition(
                name=name,
                transform=str(recommendation.get("recommended_transform", "identity")),
                expression=str(recommendation.get("expression", "x")),
                type_group=_classify_pollutant_type(name),
                observation_count=observation_count,
                importance_group=_importance_group(observation_count, settings),
            )
        )
    return pollutants


def build_land_cover_catalog(
    settings: SensorAnalysisSettings,
) -> list[LandCoverSubclassDefinition]:
    """Return the stable land-cover subclass catalog."""
    return [
        LandCoverSubclassDefinition(
            identifier=subclass,
            label=settings.subclass_labels.get(subclass, subclass),
        )
        for subclass in settings.land_cover_subclasses
    ]


__all__ = [
    "LandCoverSubclassDefinition",
    "PollutantDefinition",
    "build_land_cover_catalog",
    "build_pollutant_catalog",
]
