# SPDX-License-Identifier: Apache-2.0
"""Deterministic medium-sized dataset for the customer-service example.

The first rows used by the hand-written scenarios remain in ``store_tools.py``.
This module adds enough catalog, order, customer, and support-document variety to
exercise MCP discovery, planning, pagination, filtering, and evidence grounding
without checking a generated SQLite database into the repository.

Knowledge articles are multi-section documents rather than single paragraphs so
that a search result must be truncated into an excerpt, which forces a grounding
loop to make a second call to read the full text.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

EXTRA_CUSTOMER_COUNT = 100
EXTRA_PRODUCT_COUNT = 200
EXTRA_ORDER_COUNT = 485

# Fixed anchor so every generated date is reproducible across machines and runs.
DATASET_ANCHOR = date(2026, 3, 2)

# Lead time in days from order placement to the promised fulfillment date.
FULFILLMENT_LEAD_DAYS = {
    "parcel": 4,
    "pickup": 1,
    "appointment": 9,
    "freight": 12,
}

# Categories whose orders are not shipped as ordinary parcels.
_APPOINTMENT_CATEGORIES = frozenset({"appliances"})
_FREIGHT_CATEGORIES = frozenset({"building materials", "flooring", "doors and windows"})


@dataclass(frozen=True)
class CategorySpec:
    name: str
    products: tuple[str, ...]
    base_price: float
    price_step: float
    features: tuple[str, ...]
    return_window: int
    restocking: str
    warranty: str
    buying_advice: str
    care_advice: str
    exclusions: str
    common_issue: str


CATEGORY_SPECS: tuple[CategorySpec, ...] = (
    CategorySpec(
        "power tools",
        ("cordless drill", "impact driver", "circular saw", "jigsaw", "orbital sander"),
        79,
        24,
        ("brushless motor", "two batteries", "LED work light", "variable speed"),
        90,
        "No fee when all batteries and chargers are returned.",
        "Three-year limited tool warranty.",
        "Match the battery platform to tools you already own and choose brushless motors for frequent use.",
        "Remove the battery before cleaning and keep vents free of sawdust.",
        "Kits missing a battery, charger, or case are refunded at the value of the parts actually returned.",
        "Customers often assume batteries are cross-compatible between brands and open the kit before checking.",
    ),
    CategorySpec(
        "hand tools",
        (
            "socket set",
            "claw hammer",
            "combination wrench set",
            "locking pliers",
            "screwdriver set",
        ),
        24,
        9,
        (
            "forged steel",
            "non-slip grip",
            "corrosion-resistant finish",
            "lifetime warranty",
        ),
        90,
        "No restocking fee.",
        "Lifetime warranty against manufacturing defects.",
        "Prefer forged steel and a comfortable grip; sets should include the sizes used by the project.",
        "Wipe tools dry after use and apply light machine oil before long storage.",
        "Incomplete sets are not accepted as a full return; individual replacement pieces run through the warranty instead.",
        "Lifetime warranty claims are often filed for wear and abuse, which the warranty does not cover.",
    ),
    CategorySpec(
        "outdoor power",
        (
            "electric mower",
            "string trimmer",
            "leaf blower",
            "hedge trimmer",
            "pressure washer",
        ),
        149,
        45,
        (
            "quiet electric motor",
            "folding handle",
            "weather-resistant housing",
            "rapid charger",
        ),
        30,
        "A 15% fee applies to equipment returned with fuel, grass, or heavy wear.",
        "Three-year residential-use warranty.",
        "Compare yard size, runtime, noise limits, and whether batteries can be shared across tools.",
        "Clean clippings after use, store batteries indoors, and winterize equipment before freezing weather.",
        "Equipment used commercially is outside the residential warranty and is not accepted for buyer-remorse returns.",
        "Runtime complaints are usually a battery sizing problem rather than a defect.",
    ),
    CategorySpec(
        "garden",
        (
            "raised garden bed",
            "soaker hose",
            "pruning shears",
            "garden cart",
            "organic potting mix",
        ),
        19,
        14,
        (
            "weather-resistant",
            "ergonomic design",
            "recycled materials",
            "easy assembly",
        ),
        60,
        "Opened soil and live goods are non-returnable unless defective.",
        "One-year warranty on manufactured goods; plants follow the live-goods guarantee.",
        "Choose materials for the local climate and size watering products to the planted area.",
        "Rinse tools after use, dry metal parts, and store soil products in a sealed dry container.",
        "Perishable live goods, opened growing media, and seasonal clearance stock are excluded.",
        "Plant failures are frequently a watering or hardiness-zone issue rather than a nursery defect.",
    ),
    CategorySpec(
        "paint",
        (
            "interior wall paint",
            "exterior house paint",
            "primer",
            "paint sprayer",
            "roller kit",
        ),
        18,
        11,
        (
            "low-VOC formula",
            "washable finish",
            "mildew resistance",
            "one-coat coverage",
        ),
        30,
        "Custom-tinted paint is final sale unless mixed incorrectly.",
        "Coverage and adhesion guarantee when preparation instructions are followed.",
        "Check surface type, sheen, square footage, and whether primer is needed before choosing paint.",
        "Seal containers tightly, protect from freezing, and clean applicators with the solvent named on the label.",
        "Tinted, opened, or frozen paint is excluded; unopened factory-packaged base and sundries are returnable.",
        "Color disappointment on a correctly mixed custom tint is the single most common return dispute.",
    ),
    CategorySpec(
        "plumbing",
        (
            "kitchen faucet",
            "toilet repair kit",
            "PEX fitting set",
            "sump pump",
            "water heater connector",
        ),
        29,
        28,
        (
            "lead-free construction",
            "tool-free installation",
            "corrosion resistance",
            "code-compliant fittings",
        ),
        90,
        "Installed or cut plumbing parts cannot be returned unless defective.",
        "Five-year limited warranty on fixtures; one year on service parts.",
        "Confirm pipe size, connection type, pressure rating, and local code requirements.",
        "Shut off water before service, inspect connections for leaks, and avoid abrasive cleaners on finishes.",
        "Parts that have carried water, been cut to length, or been sealed with compound are excluded.",
        "Leaks after installation are usually a sealing or torque error rather than a defective fixture.",
    ),
    CategorySpec(
        "electrical",
        (
            "circuit breaker",
            "GFCI outlet",
            "extension cord",
            "wire spool",
            "surge protector",
        ),
        12,
        16,
        ("UL listed", "tamper-resistant", "outdoor rated", "overload protection"),
        90,
        "Cut wire and opened breaker multipacks are non-returnable unless defective.",
        "Two-year limited electrical warranty.",
        "Match voltage, amperage, wire gauge, environment rating, and panel compatibility exactly.",
        "Disconnect power before service and inspect cords and terminals regularly for heat or damage.",
        "Wire cut from a spool, opened multipacks, and any device showing arcing or heat damage are excluded.",
        "Breakers bought for the wrong panel brand are a frequent and non-returnable mistake once opened.",
    ),
    CategorySpec(
        "lighting",
        (
            "LED ceiling fixture",
            "pendant light",
            "under-cabinet light",
            "outdoor wall lantern",
            "smart bulb pack",
        ),
        24,
        22,
        (
            "dimmable LEDs",
            "warm-to-daylight color",
            "Energy Star certification",
            "wet-location rating",
        ),
        90,
        "No fee when mounting hardware and shades are complete.",
        "Five-year LED warranty; two years on fixtures.",
        "Compare lumens rather than watts and verify dimmer, ceiling-box, and wet-location compatibility.",
        "Turn off power before cleaning and use a dry microfiber cloth on bulbs and diffusers.",
        "Fixtures missing shades, canopies, or mounting hardware, and any hard-wired unit showing installation marks, are excluded.",
        "Flicker and buzzing complaints are almost always dimmer incompatibility, not a failed fixture.",
    ),
    CategorySpec(
        "flooring",
        (
            "luxury vinyl plank",
            "laminate flooring",
            "porcelain tile",
            "engineered hardwood",
            "floor underlayment",
        ),
        32,
        17,
        (
            "water-resistant core",
            "scratch-resistant finish",
            "click-lock installation",
            "low-VOC certification",
        ),
        90,
        "Unopened full cartons may be returned; opened cartons and cut material are non-returnable.",
        "Twenty-year residential finish warranty when installed to specification.",
        "Order from one dye lot, include ten percent waste, and check subfloor and moisture requirements.",
        "Sweep regularly, use manufacturer-approved cleaner, and wipe standing water promptly.",
        "Opened cartons, cut planks, special-order dye lots, and discontinued runs are excluded.",
        "Shade variation between dye lots on a reorder is the most common post-installation complaint.",
    ),
    CategorySpec(
        "hardware",
        (
            "door hinge pack",
            "cabinet pull set",
            "deck screw box",
            "wall anchor kit",
            "padlock",
        ),
        9,
        8,
        (
            "stainless steel",
            "weather-resistant coating",
            "load-rated design",
            "matching fasteners",
        ),
        90,
        "No restocking fee for complete packages.",
        "Limited lifetime mechanical warranty.",
        "Choose the correct material, finish, dimensions, and load rating for the application.",
        "Keep moving parts clean, lubricate when specified, and replace corroded fasteners.",
        "Loose fasteners sold by weight and opened packages missing pieces are excluded.",
        "Finish mismatch between production runs surprises customers extending an earlier project.",
    ),
    CategorySpec(
        "storage",
        (
            "garage cabinet",
            "steel shelving unit",
            "storage tote set",
            "wall track system",
            "tool chest",
        ),
        39,
        37,
        (
            "adjustable shelves",
            "lockable doors",
            "high load capacity",
            "modular expansion",
        ),
        90,
        "A 10% fee applies to assembled storage furniture without packaging.",
        "Five-year structural warranty.",
        "Measure the space and compare total load capacity, shelf spacing, mobility, and anchoring requirements.",
        "Anchor tall units, distribute weight evenly, and keep drawer slides clean.",
        "Assembled units with drilled or modified panels are excluded from return.",
        "Shelf sag reports usually trace to exceeding the per-shelf rather than the total load rating.",
    ),
    CategorySpec(
        "appliances",
        (
            "front-load washer",
            "French-door refrigerator",
            "convection range",
            "dishwasher",
            "over-range microwave",
        ),
        449,
        180,
        (
            "Energy Star certified",
            "quiet operation",
            "smart diagnostics",
            "stainless finish",
        ),
        48,
        "Major appliances must be reported damaged within 48 hours; buyer-remorse returns carry a 25% fee.",
        "One-year parts and labor warranty with extended compressor coverage where listed.",
        "Verify dimensions, door swing, hookups, voltage, delivery path, and installation requirements.",
        "Clean filters and seals on schedule and leave ventilation clear around the appliance.",
        "Installed, used, or customer-transported appliances are excluded from buyer-remorse return.",
        "Refused deliveries caused by a doorway or stairwell that was never measured are the top delivery failure.",
    ),
    CategorySpec(
        "kitchen",
        (
            "pull-down faucet",
            "stainless sink",
            "garbage disposal",
            "range hood",
            "butcher-block countertop",
        ),
        69,
        52,
        (
            "food-safe finish",
            "sound insulation",
            "easy-clean surface",
            "installation hardware included",
        ),
        90,
        "Cut countertops and installed sinks are non-returnable unless defective.",
        "Limited lifetime finish warranty on sinks and faucets; two years on powered products.",
        "Confirm cabinet width, cutout dimensions, drain position, airflow, and finish compatibility.",
        "Use non-abrasive cleaner, dry standing water, and renew wood finishes as directed.",
        "Cut, drilled, or sealed countertop material and any installed sink or disposal are excluded.",
        "Sink cutouts made to the wrong template are a frequent non-returnable loss.",
    ),
    CategorySpec(
        "bath",
        (
            "bath vanity",
            "dual-flush toilet",
            "showerhead",
            "medicine cabinet",
            "towel bar set",
        ),
        39,
        58,
        (
            "WaterSense certified",
            "moisture-resistant finish",
            "soft-close hardware",
            "easy-clean coating",
        ),
        90,
        "Installed sanitary fixtures are non-returnable unless defective.",
        "One-year parts warranty; ceramic fixtures carry a limited lifetime warranty.",
        "Measure rough-in, wall space, plumbing locations, and door clearance before purchase.",
        "Use non-abrasive bathroom cleaner and inspect caulk and supply connections periodically.",
        "Installed toilets, cut vanity tops, and special-order finishes are excluded.",
        "Toilets bought without checking the rough-in measurement rarely fit the existing drain.",
    ),
    CategorySpec(
        "doors and windows",
        (
            "fiberglass entry door",
            "storm door",
            "sliding patio door",
            "double-hung window",
            "window screen kit",
        ),
        79,
        115,
        ("low-E glass", "insulated core", "weather stripping", "reversible handing"),
        90,
        "Custom-sized units and installed products are non-returnable.",
        "Ten-year limited frame warranty and twenty-year insulated-glass warranty.",
        "Measure the rough opening in three places and verify handing, jamb depth, climate rating, and code.",
        "Clean tracks, lubricate hardware with dry lubricant, and inspect weather seals each season.",
        "Made-to-measure units, trimmed frames, and installed products are excluded.",
        "Handing is ordered backwards often enough that installers check it before accepting delivery.",
    ),
    CategorySpec(
        "building materials",
        (
            "drywall panel",
            "pressure-treated lumber",
            "concrete mix",
            "roofing shingle bundle",
            "insulation roll",
        ),
        11,
        13,
        (
            "code-compliant rating",
            "moisture resistance",
            "contractor pack",
            "documented coverage",
        ),
        90,
        "Cut lumber, mixed compounds, and weather-exposed materials are non-returnable.",
        "Manufacturer warranty varies by product and installation system.",
        "Calculate dimensions plus waste and verify structural, fire, moisture, and local-code requirements.",
        "Store flat, covered, and off the ground; follow temperature and moisture limits before installation.",
        "Cut, mixed, wet, or weather-damaged material and special-order quantities are excluded.",
        "Overordered bulk material stored outdoors usually fails the condition check on return.",
    ),
    CategorySpec(
        "safety",
        (
            "safety glasses",
            "hearing protection",
            "respirator kit",
            "work gloves",
            "hard hat",
        ),
        8,
        12,
        (
            "ANSI rated",
            "adjustable fit",
            "replaceable filters",
            "high-visibility details",
        ),
        30,
        "Opened respirator cartridges and worn protective equipment are non-returnable.",
        "One-year manufacturing-defect warranty.",
        "Select protection certified for the specific hazard and confirm fit before beginning work.",
        "Inspect before every use, replace damaged parts, and store clean and dry away from chemicals.",
        "Any worn personal protective equipment and opened filter cartridges are excluded for hygiene reasons.",
        "Respirators are frequently bought without checking the cartridge rating for the actual hazard.",
    ),
    CategorySpec(
        "cleaning",
        (
            "wet dry vacuum",
            "steam mop",
            "shop broom",
            "pressure sprayer",
            "concentrated cleaner",
        ),
        14,
        27,
        (
            "HEPA filtration",
            "low-noise motor",
            "refillable tank",
            "biodegradable formula",
        ),
        30,
        "Opened chemicals are non-returnable unless leaking or defective.",
        "Two-year appliance warranty.",
        "Match the product to the surface, debris type, ventilation, and any chemical restrictions.",
        "Empty tanks after use, clean filters, and never mix cleaning chemicals.",
        "Opened concentrates and any equipment returned with residue or debris in the tank are excluded.",
        "Suction complaints are almost always a clogged filter rather than a motor fault.",
    ),
    CategorySpec(
        "heating and cooling",
        (
            "portable air conditioner",
            "space heater",
            "ceiling fan",
            "air purifier",
            "smart thermostat",
        ),
        49,
        48,
        (
            "Energy Star certified",
            "programmable controls",
            "quiet mode",
            "overheat protection",
        ),
        30,
        "Seasonal products returned after use carry a 15% fee unless defective.",
        "Three-year limited warranty.",
        "Size equipment to the room, check electrical load, noise, filtration, and installation clearance.",
        "Replace filters, keep air paths clear, and inspect cords and drains at the start of each season.",
        "End-of-season returns of used seasonal equipment and installed hard-wired units are excluded.",
        "Undersized units bought for a large room get reported as defective when the room never cools.",
    ),
    CategorySpec(
        "smart home",
        (
            "video doorbell",
            "smart lock",
            "security camera",
            "smart thermostat",
            "lighting hub",
        ),
        59,
        34,
        (
            "Matter compatibility",
            "encrypted local storage",
            "voice-assistant support",
            "two-factor authentication",
        ),
        30,
        "Devices must be factory-reset and include all mounting parts.",
        "Two-year limited electronics warranty.",
        "Check Wi-Fi band, hub requirements, platform compatibility, subscription terms, and privacy controls.",
        "Install firmware updates, use unique passwords, and factory-reset devices before transfer or return.",
        "Devices still bound to a previous account and any unit missing mounting hardware are excluded.",
        "Setup failures are usually a 5 GHz-only network where the device needs 2.4 GHz.",
    ),
)

BRANDS = (
    "Atlas",
    "Beacon",
    "Cedar",
    "Forge",
    "Harbor",
    "Northstar",
    "Oakline",
    "Pioneer",
    "Summit",
    "Westfield",
)
FIRST_NAMES = (
    "Avery",
    "Jordan",
    "Morgan",
    "Riley",
    "Taylor",
    "Cameron",
    "Casey",
    "Drew",
    "Emerson",
    "Finley",
    "Hayden",
    "Jamie",
    "Kai",
    "Logan",
    "Parker",
    "Quinn",
    "Reese",
    "Robin",
    "Sage",
    "Skyler",
)
LAST_NAMES = (
    "Anderson",
    "Bennett",
    "Chen",
    "Diaz",
    "Edwards",
    "Foster",
    "Gupta",
    "Harris",
    "Ivanov",
    "Jackson",
    "Khan",
    "Lewis",
    "Martin",
    "Nguyen",
    "Owens",
    "Patel",
    "Rivera",
    "Singh",
    "Turner",
    "Williams",
)


def generated_customers() -> list[tuple[str, str, str, str]]:
    rows = []
    for offset in range(EXTRA_CUSTOMER_COUNT):
        number = offset + 9
        first = FIRST_NAMES[offset % len(FIRST_NAMES)]
        last = LAST_NAMES[(offset * 7) % len(LAST_NAMES)]
        tier = ("standard", "standard", "premium", "pro")[offset % 4]
        rows.append(
            (
                f"C-{number:03d}",
                f"{first} {last}",
                f"customer{number:03d}@example.com",
                tier,
            )
        )
    return rows


def generated_verification_codes() -> list[tuple[str, str]]:
    return [
        (f"C-{number:03d}", f"{(number * 7919) % 1_000_000:06d}")
        for number in range(9, 9 + EXTRA_CUSTOMER_COUNT)
    ]


def generated_products() -> list[tuple[str, str, float, str, int, str]]:
    rows = []
    product_number = 26
    for spec in CATEGORY_SPECS:
        for variant in range(10):
            brand = BRANDS[variant]
            product = spec.products[variant % len(spec.products)]
            feature_a = spec.features[variant % len(spec.features)]
            feature_b = spec.features[(variant + 1) % len(spec.features)]
            name = f"{brand} {product.title()} {variant + 1}"
            price = round(spec.base_price + spec.price_step * variant, 2)
            in_stock = 0 if variant in (3, 8) else 1
            description = (
                f"{name} is a {spec.name} product with {feature_a} and {feature_b}. "
                f"Designed for residential projects and backed by {spec.warranty.lower()}"
            )
            rows.append(
                (
                    f"P-{product_number:03d}",
                    name,
                    price,
                    spec.name,
                    in_stock,
                    description,
                )
            )
            product_number += 1
    assert len(rows) == EXTRA_PRODUCT_COUNT
    return rows


def fulfillment_method(category: str, offset: int) -> str:
    """Pick a delivery method from the product category, deterministically."""
    if category in _APPOINTMENT_CATEGORIES:
        return "appointment"
    if category in _FREIGHT_CATEGORIES:
        return "freight"
    return "pickup" if offset % 7 == 0 else "parcel"


# Order status cycle. Kept as the driver of the mix so the dataset holds a
# realistic spread; the dates are then derived from it rather than cycled
# independently.
_ORDER_STATUSES = (
    "delivered",
    "shipped",
    "processing",
    "pending",
    "delivered",
    "cancelled",
)


def order_age_days(status: str, lead_days: int, spread: int) -> int:
    """Days between the order date and the anchor, consistent with the status.

    Status and age used to be two independent cycles, which produced orders
    that contradicted themselves: one placed six months ago still "pending",
    and one placed yesterday already "delivered" with a promised date still in
    the future. An agent reading those cannot answer "where is my order"
    sensibly, and neither can a scenario grounded in them.

    The promised date is the order date plus the lead time, so age relative to
    the lead time is what decides which stage an order can be in.
    """
    if status in ("pending", "processing"):
        # Not yet due: still inside its lead time.
        return spread % max(1, lead_days)
    if status == "shipped":
        # In transit, arriving around now. A day late is normal and is the case
        # a customer asks about.
        return lead_days + (spread % 2)
    # Delivered and cancelled are both settled, and settled well before now.
    return lead_days + 5 + spread


def generated_orders(
    products: list[tuple[str, str, float, str, int, str]],
) -> list[tuple[str, str, str, float, str, str, str, str]]:
    rows = []
    for offset in range(EXTRA_ORDER_COUNT):
        number = offset + 1016
        customer_number = 9 + (offset % EXTRA_CUSTOMER_COUNT)
        product = products[(offset * 17) % len(products)]
        quantity = 1 + (offset % 3)
        total = round(product[2] * quantity, 2)
        method = fulfillment_method(product[3], offset)
        status = _ORDER_STATUSES[offset % len(_ORDER_STATUSES)]
        lead = FULFILLMENT_LEAD_DAYS[method]
        # Varies the dates within a status without breaking its invariant.
        spread = (offset // len(_ORDER_STATUSES) * 3) % 120
        created = DATASET_ANCHOR - timedelta(days=order_age_days(status, lead, spread))
        promised = created + timedelta(days=lead)
        rows.append(
            (
                f"ORD-{number}",
                f"C-{customer_number:03d}",
                status,
                total,
                f"{product[1]} x{quantity}",
                created.isoformat(),
                method,
                promised.isoformat(),
            )
        )
    return rows


# ── Knowledge articles ──
#
# Each article is a multi-section document. Search returns a short excerpt, so
# grounding on the full policy requires a follow-up read call.


def _return_policy_article(spec: CategorySpec) -> str:
    return "\n\n".join(
        (
            f"# {spec.name.title()} return policy",
            "## Return window\n"
            f"{spec.name.title()} products may be returned within "
            f"{spec.return_window} days with proof of purchase. The window runs "
            "from the delivery date for shipped orders and from the collection "
            "date for in-store pickup. Pro accounts purchasing on contract terms "
            "may have a different window recorded on the quote.",
            f"## Condition and fees\n{spec.restocking} Products must include "
            "their original components, manuals, and packaging, and must not "
            "show misuse. A return that arrives incomplete may be refused, or "
            "refunded at the value of the parts actually returned.",
            f"## Exclusions\n{spec.exclusions} Clearance and final-sale items "
            "identified as such at checkout cannot be returned unless they are "
            "defective.",
            "## Defective products\n"
            f"Defective products follow {spec.warranty.lower()} Report the "
            "failure with the order number, the model and serial number, and "
            "photographs where the fault is visible. Do not disassemble powered "
            "equipment before support responds, because doing so can void "
            "coverage.",
            f"## Frequent disputes\n{spec.common_issue} Agents should confirm "
            "the purchase date, the condition of the item, and whether an "
            "exclusion applies before promising a refund.",
        )
    )


def _buying_guide_article(spec: CategorySpec) -> str:
    return "\n\n".join(
        (
            f"# Choosing {spec.name}",
            f"## What to compare\nWhen choosing {spec.name}, "
            f"{spec.buying_advice} Compare the product description, "
            "compatibility notes, stock state, and warranty before purchase.",
            "## Features that matter\n"
            f"Listings in this category describe features such as "
            f"{', '.join(spec.features[:-1])}, and {spec.features[-1]}. Treat "
            "these as a shortlist filter rather than a ranking: the right "
            "choice depends on how often the product is used and the conditions "
            "it is used in.",
            "## Budget and availability\n"
            f"Prices in {spec.name} span a wide range, and out-of-stock items "
            "stay listed so that customers can compare them. Confirm stock "
            "before recommending a specific model, and offer an in-stock "
            "alternative in the same category when the preferred item is "
            "unavailable.",
            f"## Where customers go wrong\n{spec.common_issue} Asking about the "
            "project, the space, and any equipment the customer already owns "
            "prevents most mismatched purchases.",
            f"## After the purchase\nReturns in this category follow the "
            f"{spec.name} return policy, which allows {spec.return_window} "
            f"days. {spec.exclusions}",
        )
    )


def _care_article(spec: CategorySpec) -> str:
    return "\n\n".join(
        (
            f"# Caring for {spec.name}",
            f"## Routine care\nFor safe use and long service life, "
            f"{spec.care_advice} Follow the product manual and discontinue use "
            "if guards, cords, batteries, or structural parts are damaged.",
            "## Storage\nStore products clean, dry, and out of direct sunlight. "
            "Extremes of temperature and humidity shorten the life of "
            "batteries, adhesives, finishes, and seals, so keep storage areas "
            "ventilated and away from freezing conditions.",
            "## Troubleshooting\n"
            f"{spec.common_issue} Before treating a problem as a defect, check "
            "that the product is being used within its rated capacity and that "
            "routine maintenance is current.",
            "## When to claim\n"
            f"Coverage is {spec.warranty.lower()} Wear parts, consumables, and "
            "damage from misuse or improper installation are not covered. Keep "
            "the receipt and the model and serial number with the product so "
            "that a claim can be filed without delay.",
        )
    )


SERVICE_ARTICLES: tuple[tuple[str, str, str, str], ...] = (
    (
        "shipping",
        "Standard, expedited, and express shipping",
        "Standard parcel shipping normally arrives in three to five business days. Expedited shipping normally arrives in two business days, and express shipping normally arrives the next business day for orders placed before noon local time. Hazardous, oversized, and special-order goods may use different services shown at checkout.",
        "Transit estimates start when the order leaves the fulfillment center, not when it is placed, and business days exclude weekends and public holidays. Remote and offshore destinations add transit time regardless of the service selected.\n\nSome goods cannot move by air. Flammable liquids, pressurized cylinders, and large lithium batteries ship by ground only, so an expedited upgrade on those items does not shorten transit. The checkout page shows the services actually available for the cart.\n\nWhen an order contains both stocked and special-order items, the stocked items ship first and each shipment carries its own tracking number.",
    ),
    (
        "shipping",
        "Free-shipping qualification",
        "Parcel orders of $45 or more qualify for free standard shipping after discounts and before tax. Gift cards, installation labor, delivery charges, and scheduled freight do not count toward the threshold. Pro members may have separate contract terms.",
        "The threshold is evaluated on the merchandise subtotal after coupons are applied, so a discount can drop an order below the threshold and reintroduce a shipping charge. The order summary recalculates this before payment.\n\nOversized and heavy items carry a separate handling surcharge that free standard shipping does not waive. That surcharge is shown on the product page and again at checkout.\n\nIf items in a qualifying order are returned so that the remaining merchandise falls below the threshold, the shipping charge is not retroactively applied.",
    ),
    (
        "delivery",
        "Scheduled appliance delivery",
        "Major-appliance delivery uses a four-hour appointment window. The customer must provide a clear path, working shutoff valves, grounded outlets, and an adult at least 18 years old. Haul-away and installation are separate line items and must be purchased before the route is dispatched.",
        "Measure every doorway, hallway, stairwell, and turn on the delivery path, including the depth of the appliance with its door and handles attached. Crews cannot remove doors, windows, railings, or cabinetry, and cannot hoist units through openings.\n\nThe crew will not connect an appliance to shutoff valves, outlets, gas lines, or venting that are damaged, corroded, or out of code. When the site is not ready the delivery is refused at the door, the appointment is rescheduled, and a redelivery fee may apply.\n\nHaul-away removes one comparable old appliance per new unit and requires it to be disconnected and emptied before arrival. Confirm the appointment window the evening before; the driver calls ahead but will not wait beyond the window.",
    ),
    (
        "delivery",
        "Freight and curbside delivery",
        "Palletized building materials are delivered curbside to an accessible hard surface. Drivers cannot cross lawns, enter homes, or use customer equipment. Hidden damage must be reported within 48 hours with photos of the product and packaging.",
        "The customer is responsible for moving material from the curb and for having enough help available at the appointment. Liftgate service must be requested when the site has no forklift or loading dock, and it is priced separately.\n\nInspect the shipment before signing. Visible damage should be noted on the delivery receipt at the time of delivery, because a clean signature makes a later visible-damage claim much harder to resolve.\n\nNarrow streets, weight-restricted roads, unpaved driveways, and low clearances can make a site unreachable by a full-size truck. Flag any of these when scheduling so the shipment can be routed on a smaller vehicle.",
    ),
    (
        "returns",
        "Returning online purchases in store",
        "Most parcel orders can be returned at any store with the order number and government-issued photo identification. Refunds go to the original payment method. Category exclusions, return windows, and restocking fees still apply.",
        "A printed or digital packing slip speeds up the return, but the order can also be located by the order number or the card used. Returns without any proof of purchase may be offered as store credit at the current selling price, subject to limits.\n\nItems delivered by freight or appointment are not accepted at a store counter; those are collected by a scheduled pickup arranged through support.\n\nA refund to a card is submitted when the return is processed in store, and the customer keeps the return receipt as evidence until the credit appears on the statement.",
    ),
    (
        "returns",
        "Damaged or missing items",
        "Report parcel damage or missing items within seven days of delivery. Keep all packaging and photograph the shipping label, outer carton, and damage. The support team may replace the item, ship missing components, or issue a refund after review.",
        "Photographs of the outer carton and the shipping label are what allow a carrier claim to be filed, so discarding the packaging before support responds can prevent a replacement from being approved.\n\nWhen a multi-piece order arrives incomplete, check the tracking numbers first: items in one order frequently ship separately and arrive on different days. Support can confirm whether an outstanding piece is still in transit.\n\nMissing components of an otherwise intact product are usually resolved by shipping the part directly from the manufacturer, which is faster than returning and reordering the whole item.",
    ),
    (
        "pricing",
        "Price-match policy",
        "The store matches the current advertised price of an identical, in-stock item sold by an authorized local or online retailer. Marketplace sellers, auctions, clearance, rebates, installation bundles, typographical errors, and membership-only prices are excluded.",
        "Identical means the same brand, model number, colour, and configuration, sold new and available to any customer at the advertised price. A different model number in the same product family does not qualify.\n\nThe competing offer must be verifiable at the time of the request, and the item must be in stock at that retailer for delivery to the customer's area. A price that appears only in a cart, after a login, or inside a membership program cannot be matched.\n\nA price match is applied once per item and cannot be combined with a further percentage-off coupon on the same item. Requests after purchase are considered within the return window and are handled as a price adjustment.",
    ),
    (
        "pricing",
        "Promotions and coupon stacking",
        "One percentage or fixed-amount coupon may be applied per order unless its terms explicitly allow stacking. Manufacturer rebates are submitted after purchase and may be combined with store discounts. Coupons do not reduce delivery, installation, tax, or gift-card charges.",
        "Bundle promotions that already discount a group of items usually exclude further coupons, and the coupon terms name the excluded categories. Special-order and custom-configured products are commonly excluded.\n\nWhen a coupon and a promotional price both apply, the system takes whichever produces the lower total rather than compounding them, and the order summary shows which discount was used.\n\nIf a coupon expires between placing an order and shipping it, the discount captured at checkout is honoured. Cancelling and reordering after expiry does not preserve it.",
    ),
    (
        "warranty",
        "Manufacturer and store warranties",
        "The manufacturer warranty covers defects for the period listed on the product. It does not cover misuse, normal wear, consumables, or improper installation. Store protection plans begin when the manufacturer warranty ends unless the plan documents say otherwise.",
        "Warranty length often differs by component within one product: a compressor, a motor, or a finish may carry longer coverage than the rest of the unit. The product documentation is the authority on which term applies.\n\nCommercial or rental use of a product warranted for residential use generally voids the manufacturer warranty, even when the product is otherwise undamaged.\n\nA protection plan is tied to the purchase and, where the plan permits transfer, may follow the product to a new owner with the original receipt.",
    ),
    (
        "warranty",
        "Starting a warranty claim",
        "A warranty claim needs proof of purchase, product model and serial number, a description of the failure, and photos when relevant. Customers should not disassemble powered equipment before receiving instructions because doing so may void coverage.",
        "The manufacturer decides between repair, replacement, and refund under most warranties, and the store cannot override that determination. What the store can do is supply duplicate receipts and purchase records.\n\nSerial numbers are on the product data plate rather than the carton, so recording the number before installation avoids having to reach an installed unit later.\n\nWhile a claim is open, keep the product available for inspection. Disposing of a failed unit before the claim is resolved usually ends the claim.",
    ),
    (
        "installation",
        "Installation service estimates",
        "Installation begins with a measurement or site assessment where required. The quote covers only the listed scope; code upgrades, concealed damage, permits, disposal, and customer-requested changes may add cost. Customers approve change orders before additional work begins.",
        "The measurement takes precedence over customer-supplied dimensions, and material is ordered from it. When the two disagree, the installer re-measures before ordering.\n\nConcealed conditions are the main source of change orders: rot, unsupported subfloor, non-compliant wiring, or asbestos-containing material cannot be seen until work opens the area. Work stops and the customer approves the revised scope before it resumes.\n\nPermits and inspections are scheduled by the installer where the jurisdiction requires them, and the timeline depends on the local authority rather than the store.",
    ),
    (
        "installation",
        "Preparing for an installer",
        "Clear furniture and personal belongings from the work area, provide parking and utility access, secure pets, and ensure an adult decision-maker is present. Products supplied by the customer must be onsite, complete, and compatible before the appointment.",
        "Installers do not move large furniture, disconnect electronics, or take down window coverings, so an unprepared room costs appointment time and can force a reschedule.\n\nWater, power, and gas must be available and functional. Where a shutoff is seized or missing, the installer may be unable to proceed until a repair is completed by the customer.\n\nAn adult who can approve a change order must be reachable for the whole appointment. Work that requires approval pauses until it is given.",
    ),
    (
        "accounts",
        "Pro account benefits",
        "Verified Pro accounts can receive volume pricing, purchase tracking, saved quotes, job-site delivery options, and delegated purchasers. Credit terms and tax-exempt status require separate approval and are not activated merely by creating an online profile.",
        "Volume pricing is quoted per job rather than applied automatically at checkout, and a saved quote holds its pricing for the period stated on the quote.\n\nDelegated purchasers buy against the account within limits the account owner sets, and each purchase is attributed to the buyer for reporting.\n\nJob-site delivery needs a site contact and access details recorded on the order, because the delivery address is frequently not a serviceable street address.",
    ),
    (
        "accounts",
        "Tax-exempt purchases",
        "Approved organizations must associate a valid exemption certificate with their account before checkout. Exemption cannot be applied to another organization or to personal purchases. Corrections after purchase require the receipt and may be limited by state rules.",
        "Certificates are state-specific: an organization buying in several states registers a certificate for each one, and an out-of-state certificate is not accepted where the purchase takes place.\n\nCertificates expire. When a certificate lapses, checkout charges tax normally until a current certificate is on file, which frequently surprises long-standing accounts.\n\nA refund of tax already charged is processed against the original transaction, and some states cap how long after the purchase that correction can be made.",
    ),
    (
        "payments",
        "Payment methods and authorization holds",
        "Online orders accept major cards, store cards, gift cards, and supported digital wallets. A temporary authorization may appear before shipment. Split tender is supported between one gift card and one additional payment method, but not between two credit cards.",
        "An authorization hold is not a charge. It reserves funds and is released by the bank, usually within a few business days, when the final capture happens at shipment.\n\nOrders that ship in several shipments are captured per shipment, so a customer sees more than one charge that together match the order total.\n\nA declined authorization holds the order rather than cancelling it immediately, and the customer is given a window to supply another payment method before the order is released.",
    ),
    (
        "payments",
        "Refund timing",
        "Card refunds are submitted when the return is processed and usually appear within three to seven business days, depending on the bank. Cash purchases over the store threshold may be refunded by mailed check. Gift-card purchases are refunded to a gift card.",
        "The store controls when a refund is submitted, not when it posts. Once a reference number exists, any further delay sits with the issuing bank, and the customer can quote that reference when asking their bank.\n\nA refund to an expired or closed card is routed by the bank to the replacement account in most cases. Where it is rejected, the refund is reissued by another method after verification.\n\nSplit-tender orders refund in reverse order of payment: the gift card portion first, then the remaining method.",
    ),
    (
        "orders",
        "Changing or cancelling an order",
        "A parcel order can be cancelled while it is processing but not after it enters picking or ships. Scheduled delivery orders may be changed until the route cutoff shown in the confirmation. Custom and made-to-order products cannot be cancelled after production begins.",
        "The window between placing an order and picking it is short, often under an hour for stocked items, so a cancellation request needs to be actioned immediately rather than queued.\n\nWhen an order has already shipped, the remaining option is a return once it arrives, or a refusal at the door for delivery orders. A refused delivery is processed as a return and follows the same category exclusions and fees.\n\nAddress changes after an order ships cannot be made by the store; the carrier may offer a redirect to the recipient directly.",
    ),
    (
        "orders",
        "Backorders and substitutions",
        "A backordered item keeps its place until inventory arrives or the customer cancels it. The store does not substitute a different model without approval. Other items in the order may ship separately and generate separate tracking numbers.",
        "Estimated arrival dates on a backorder come from the supplier and move as the supplier updates them. The customer is notified when the estimate changes materially.\n\nA backordered item is not charged until it ships, so the authorization for it may be reauthorized as the date moves.\n\nWhere a model is discontinued while on backorder, support offers the closest current equivalent and its price difference, and the customer chooses between the substitution and a cancellation.",
    ),
    (
        "rentals",
        "Tool rental deposits and returns",
        "Tool rentals require government-issued identification and an authorized payment card. The authorization amount depends on the tool. Equipment must be returned clean, fueled as instructed, and before the due time to avoid cleaning, fuel, damage, or overtime charges.",
        "Rental time is counted in clock hours from pickup, not in hours of actual use, and it continues over nights and weekends unless the rate plan says otherwise.\n\nReturning equipment dirty triggers a cleaning charge that is usually higher than the cost of cleaning it onsite. Concrete, paint, and adhesive residue are the most commonly charged.\n\nA tool that fails during the rental should be reported immediately. Continuing to use a failing tool can convert a warranty issue into billable damage.",
    ),
    (
        "rentals",
        "Rental damage protection",
        "Optional damage protection covers accidental damage from normal use, subject to the rental agreement. It does not cover theft, loss, misuse, prohibited materials, transport damage, or failure to follow operating instructions.",
        "Protection is purchased at the start of the rental and cannot be added after damage occurs. It reduces but does not always eliminate the customer's liability, and the agreement states any remaining amount.\n\nTheft and loss are the most common declined claims. A police report does not convert an excluded loss into a covered one.\n\nDamage found at return is documented with photographs before any charge is raised, and the customer may review that documentation.",
    ),
    (
        "privacy",
        "Customer data and identity verification",
        "Agents verify identity before disclosing order, address, payment, loyalty, or account details. Verification codes are single-purpose and must never be requested through public comments. Product browsing and general policy questions do not require identity verification.",
        "Verification protects the account holder, so it is required even when the caller knows the order number: an order number is not a secret and frequently appears on packaging.\n\nA code is sent to the address already on the account and is never read back to the customer by the agent. An agent who cannot verify the caller may describe policy in general terms but must not confirm or deny specific account details.\n\nRequests to change the email or phone on an account are treated as high risk and require verification against the existing contact details, not the new ones.",
    ),
    (
        "privacy",
        "Deleting an online account",
        "An account-deletion request removes the online profile after identity verification. Transaction, warranty, fraud-prevention, and legally required records may be retained. Active orders, rentals, claims, or balances must be resolved first.",
        "Deletion is not reversible, and it ends access to order history, saved quotes, protection plan records, and any stored payment methods.\n\nRetained records are held for the period the applicable law requires and are not used for marketing after deletion.\n\nWhere the customer wants to stop marketing contact rather than lose their history, an unsubscribe or a marketing-preferences change achieves that without deleting the account.",
    ),
    (
        "rebates",
        "Manufacturer rebate submissions",
        "Rebates are submitted to the manufacturer with the completed form, dated receipt, model and serial details, and any required package codes. Submission deadlines are strict. The store cannot replace a denied rebate but can provide duplicate receipts.",
        "Package codes are frequently printed on the carton rather than the product, so the carton should be kept until the rebate is submitted and acknowledged.\n\nProcessing typically takes several weeks and is handled entirely by the manufacturer or its rebate processor. Status enquiries go to the processor named on the form.\n\nA rebate is calculated on the price paid after discounts, so stacking a coupon can reduce the rebate amount.",
    ),
    (
        "gift cards",
        "Gift card restrictions",
        "Gift cards do not expire and cannot be redeemed for cash except where required by law. They cannot purchase other gift cards or pay a credit account. Lost cards can be replaced only with the original receipt and remaining balance verification.",
        "A gift card is treated like cash. Without the original receipt or a registration on an account, a lost or stolen card cannot be traced or replaced.\n\nBalances can be checked in store or online with the card number, and a card with a small remaining balance can be combined with one other payment method at checkout.\n\nA purchase made with a gift card refunds to a gift card, which surprises customers expecting the refund on a card they later used for part of the order.",
    ),
    (
        "orders",
        "Tracking an order and delivery milestones",
        "Order tracking shows the order date, the promised date, and the current fulfillment stage. Parcel orders expose a carrier tracking number once they ship; appointment and freight orders expose a scheduled window instead. The promised date is an estimate, not a guarantee.",
        "Stages run from pending to processing, then to shipped and finally delivered. A cancelled order stops at the point where it was cancelled and shows no further movement.\n\nCarrier tracking can lag physical movement by several hours, and a parcel occasionally shows as delivered before the driver completes the scan. Waiting until the end of the delivery day before opening a missing-parcel case avoids most false reports.\n\nWhere a promised date has passed with no movement, support can open a trace with the carrier. A trace typically takes a few business days to resolve.",
    ),
    (
        "pickup",
        "Buy online, pick up in store",
        "Pickup orders are usually ready within two hours during store hours and are held for seven days. The person collecting must show government-issued photo identification and the order number. An alternate pickup person can be named on the order before it is collected.",
        "A held order that is not collected within the hold period is returned to stock and refunded automatically to the original payment method.\n\nOnly the named purchaser or an alternate recorded on the order can collect it, and identification is checked at the counter for every pickup regardless of order value.\n\nItems that are not stocked at the selected store are transferred in, and the ready notification is sent when the transfer arrives rather than when the order is placed.",
    ),
    (
        "special orders",
        "Special-order and custom products",
        "Special-order and custom-configured products are made or sourced for one customer and cannot be cancelled once production or the supplier order begins. A deposit is collected at ordering. Lead times come from the supplier and are estimates.",
        "Because these products are non-cancellable and non-returnable, every specification is confirmed with the customer in writing before the order is placed.\n\nDeposits are applied to the final balance and are forfeited where the customer cancels after production begins. The remaining balance is due before delivery or installation.\n\nA supplier delay does not create a right to cancel, but support will confirm a revised date and can escalate where the delay is substantial.",
    ),
    (
        "safety notices",
        "Product recalls and safety notices",
        "A recalled product should be taken out of service immediately and not returned to a store counter unless the notice says to. Recall notices specify the remedy, which may be a repair kit, a replacement, or a refund, and they are administered by the manufacturer.",
        "Model and serial ranges in a notice are precise, so a product from the same family is not necessarily affected. Check the data plate against the notice before acting.\n\nWhere the remedy is a refund, it is issued at the price paid with proof of purchase, and the store can supply duplicate receipts from the purchase record.\n\nA safety notice that stops short of a recall usually adds a usage restriction or a maintenance step rather than removing the product from service.",
    ),
    (
        "support",
        "Accessibility and language support",
        "Support is available by phone, chat, and in store. Interpretation is available on request in the supported languages, and accessible service options include assistance in store and alternative formats for documentation. Requesting support does not require an online account.",
        "An authorized representative may act for a customer where the customer has recorded that authorization on the account, and the same identity verification applies to the representative.\n\nDocumentation such as receipts, manuals, and warranty terms can be provided in large print or a screen-reader-friendly format on request.\n\nWait times are longest at opening and after promotional events; chat generally connects faster than phone during those periods.",
    ),
)


def knowledge_articles() -> list[tuple[str, str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str, str]] = []
    number = 1
    for spec in CATEGORY_SPECS:
        slug = spec.name.replace(" ", "-")
        rows.append(
            (
                f"KB-{number:04d}",
                f"{spec.name.title()} return policy",
                "returns",
                spec.name,
                _return_policy_article(spec),
                f"store://knowledge/{slug}/returns",
            )
        )
        number += 1
        rows.append(
            (
                f"KB-{number:04d}",
                f"Choosing {spec.name}",
                "buying guide",
                spec.name,
                _buying_guide_article(spec),
                f"store://knowledge/{slug}/buying-guide",
            )
        )
        number += 1
        rows.append(
            (
                f"KB-{number:04d}",
                f"Caring for {spec.name}",
                "product care",
                spec.name,
                _care_article(spec),
                f"store://knowledge/{slug}/care",
            )
        )
        number += 1
    for topic, title, summary, detail in SERVICE_ARTICLES:
        slug = title.lower().replace(" ", "-").replace("/", "-")
        rows.append(
            (
                f"KB-{number:04d}",
                title,
                topic,
                "",
                f"# {title}\n\n{summary}\n\n{detail}",
                f"store://knowledge/service/{slug}",
            )
        )
        number += 1
    return rows


KNOWLEDGE_ARTICLE_COUNT = len(CATEGORY_SPECS) * 3 + len(SERVICE_ARTICLES)
