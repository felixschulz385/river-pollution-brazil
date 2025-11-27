# ==============================================================================
# HEALTH DATA ASSEMBLY
# ==============================================================================

# ------------------------------------------------------------------------------
# Libraries
# ------------------------------------------------------------------------------

library(tidyverse)
library(readxl)
library(sf)
library(zoo)
library(arrow)
library(spdep)
library(Matrix)

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

# Cumulative mean that ignores NA values
cummean_na <- function(x) {
    not_na <- !is.na(x)
    cumsum_x <- cumsum(ifelse(not_na, x, 0))
    cumcount_x <- cumsum(not_na)
    cummean_x <- cumsum_x / cumcount_x
    return(cummean_x)
}

# Check if panel data is balanced
check_balance <- function(data, group_var, time_var) {
    balanced_check <- data %>%
        group_by({{ group_var }}, {{ time_var }}) %>%
        summarise(count = n(), .groups = "drop") %>%
        pivot_wider(names_from = {{ time_var }}, values_from = count)
    return(!any(is.na(balanced_check)))
}

# Get municipalities with complete observations for a variable set
get_full_panel_muns <- function(data, variable_set) {
    tmp <- data %>%
        group_by(CC_2r) %>%
        summarise(across(all_of(variable_set), ~ (is.na(.) | is.infinite(.)) %>%
            `!`() %>%
            sum()))
    
    threshold <- tmp %>%
        summarise(across(where(is.numeric), median)) %>%
        t() %>%
        min()
    
    tmp %>%
        filter(if_all(all_of(variable_set), ~ . >= threshold)) %>%
        pull(CC_2r)
}

# Extract upstream weights for a given municipality
extraction_worker <- function(CC_2r) {
    tmp <- weights[CC_2r, ]
    tmp[tmp > 0]
}

# Calculate weighted sum for all variables
weighted_sum_worker <- function(variable, upstream_weights, CC_2r) {
    map_dbl(upstream_weights, ~ weighted.mean(variable[match(names(.x), CC_2r)], .x, na.rm = TRUE))
}

# ==============================================================================
# DATA LOADING
# ==============================================================================

# ------------------------------------------------------------------------------
# Health Data
# ------------------------------------------------------------------------------

mortality <- read_parquet("data/mortality/mortality_panel.parquet") %>%
    rename(CC_2r = mun_id) %>%
    mutate(CC_2r = as.character(CC_2r))

hospitalizations <- read_parquet("data/health/hospitalizations.parquet") %>%
    mutate(CC_2r = as.character(CC_2r))

birth_weight <- read_parquet("data/health/birth_weight.parquet") %>%
    mutate(CC_2r = as.character(mun_id))

# ------------------------------------------------------------------------------
# Population & Demographics
# ------------------------------------------------------------------------------

population <- read_csv("data/misc/raw/population.csv") %>%
    reframe(year = ano, CC_2r = id_municipio %>% as.character() %>% str_sub(1, 6), population = populacao)

births <- read_csv("data/mortality/raw/births.csv") %>%
    reframe(year = ano, CC_2r = id_municipio_nascimento %>% paste() %>% str_sub(1, 6), total_births)

# ------------------------------------------------------------------------------
# Environmental Data
# ------------------------------------------------------------------------------

deforestation <- read_parquet("data/land_cover/deforestation_municipalities.parquet")

climate <- read_parquet("data/climate/climate_data.parquet") %>%
    mutate(CC_2r = as.character(CC_2r))

# ------------------------------------------------------------------------------
# Control Variables
# ------------------------------------------------------------------------------

control_variables <- read_parquet("data/misc/control_variables.parquet") %>%
    mutate(CC_2r = as.character(CC_2r))

# ------------------------------------------------------------------------------
# Geographic Data
# ------------------------------------------------------------------------------

municipalities <- st_read("data/misc/raw/gadm/gadm41_BRA_2.json") %>%
    mutate(CC_2r = CC_2 %>% as.character() %>% str_sub(1, 6))

municipalities_simplified <- municipalities %>%
    st_make_valid() %>%
    st_simplify(dTolerance = 0.01)

legal_amazon <- read_excel("data/misc/raw/Municipios_da_Amazonia_Legal_2022.xlsx") %>%
    reframe(CC_2r = str_sub(CD_MUN, 1, 6)) %>%
    pull()

immediate_regions <- read_excel("data/misc/raw/regioes_geograficas_composicao_por_municipios_2017_20180911.xlsx") %>%
    reframe(CC_2r = str_sub(CD_GEOCODI, 1, 6), CC_i = str_sub(cod_rgi, 1, 6))

regions <- read_csv("data/misc/raw/brazil_regions_states.csv") %>%
    inner_join(municipalities %>% st_drop_geometry() %>% select(CC_2r, NAME_1), by = c("state" = "NAME_1")) %>%
    select(-state) %>%
    distinct()

# ------------------------------------------------------------------------------
# Spatial Weights
# ------------------------------------------------------------------------------

weights <- readMM("data/river_network/processed/weights_matrix_exponential_100.mtx")
weights_municipalities <- read_csv("data/river_network/processed/weights_municipalities.csv")
rownames(weights) <- weights_municipalities$cc_2r
colnames(weights) <- weights_municipalities$cc_2r

# ==============================================================================
# DATA PREPROCESSING
# ==============================================================================

# ------------------------------------------------------------------------------
# Mortality Rates
# ------------------------------------------------------------------------------

# Total mortality
mortality_yy <- mortality %>%
    filter(age_group %in% c("total")) %>%
    group_by(CC_2r, year) %>%
    summarise(deaths = sum(deaths, na.rm = TRUE)) %>%
    ungroup() %>%
    full_join(population, by = c("CC_2r", "year")) %>%
    group_by(CC_2r) %>%
    mutate(
        mortality_rate_tot = deaths / population,
        mortality_rate_tot_5y = rollsum(deaths, k = 5, fill = NA, align = "right") / population
    ) %>%
    ungroup()

# Child mortality (under 5)
mortality_l5 <- mortality %>%
    filter(age_group %in% c("under_1", "1_to_4")) %>%
    group_by(CC_2r, year) %>%
    summarise(deaths = sum(deaths, na.rm = TRUE)) %>%
    ungroup() %>%
    full_join(population, by = c("CC_2r", "year")) %>%
    group_by(CC_2r) %>%
    mutate(
        mortality_rate_l5 = deaths / population,
        mortality_rate_l5_5y = rollsum(deaths, k = 5, fill = NA, align = "right") / population
    ) %>%
    ungroup()

# Infant mortality (under 1)
mortality_l1 <- mortality %>%
    filter(age_group %in% c("under_1")) %>%
    group_by(CC_2r, year) %>%
    summarise(deaths = sum(deaths, na.rm = TRUE)) %>%
    ungroup() %>%
    full_join(births, by = c("CC_2r", "year")) %>%
    mutate(total_births = replace_na(total_births, 0)) %>%
    group_by(CC_2r) %>%
    mutate(
        mortality_rate_l1 = deaths / total_births * 1000,
        mortality_rate_l1_5y = rollsum(deaths, k = 5, fill = NA, align = "right") / rollsum(total_births, k = 5, fill = NA, align = "right")
    ) %>%
    ungroup()

# Combined mortality data
mortality_full <- mortality_yy %>%
    full_join(mortality_l5, by = c("CC_2r", "year")) %>%
    full_join(mortality_l1, by = c("CC_2r", "year")) %>%
    select(CC_2r, year, mortality_rate_tot, mortality_rate_tot_5y, mortality_rate_l5, mortality_rate_l5_5y, mortality_rate_l1, mortality_rate_l1_5y)

# ------------------------------------------------------------------------------
# Hospitalization & Expenditure
# ------------------------------------------------------------------------------

hospitalizations <- hospitalizations %>%
    left_join(population, by = c("CC_2r", "year")) %>%
    reframe(CC_2r, year, hosp_rate = hospitalizations / population, ex_pop = total_value / population)

# ------------------------------------------------------------------------------
# Birth Weight
# ------------------------------------------------------------------------------

birth_weight <- birth_weight %>%
    reframe(
        CC_2r, 
        year, 
        low_birth_weight = `Menos de 500g` + `500 a 999g` + `1000 a 1499 g` + `1500 a 2499 g` + `2500 a 2999 g`,
        low_birth_weight_share = low_birth_weight / Total
        ) %>%
        filter(year != 1994)

# ------------------------------------------------------------------------------
# Control Variables
# ------------------------------------------------------------------------------

control_variables <- control_variables %>%
    left_join(population, by = c("CC_2r", "year")) %>%
    arrange(CC_2r, year) %>%
    mutate(
        CC_2r = as.character(CC_2r),
        gdp_pc = gdp / population,
        urban_share = urban_population / population,
        clean_water_share = urban_population_served_water / urban_population
    ) %>%
    select(CC_2r, year, gdp_pc, clean_water_share, educ_ideb, vaccination_index_5y, health_primary_care_coverage, health_doctors_1000, urban_share, clean_water_share)

# ==============================================================================
# ANALYSIS DATA COMPILATION
# ==============================================================================

# ------------------------------------------------------------------------------
# Main Analysis Dataset
# ------------------------------------------------------------------------------

analysis <- mortality_full %>%
    full_join(., hospitalizations, by = c("CC_2r", "year")) %>%
    full_join(., birth_weight, by = c("CC_2r", "year")) %>%
    full_join(., deforestation, by = c("CC_2r", "year")) %>%
    left_join(., climate, by = c("CC_2r", "year")) %>%
    left_join(., control_variables, by = c("CC_2r", "year")) %>%
    left_join(., immediate_regions, by = "CC_2r") %>%
    left_join(., regions, by = "CC_2r") %>%
    arrange(CC_2r, year) %>%
    drop_na(CC_2r) %>%
    mutate(region_year = paste(CC_i, year, sep = "_")) %>%
    group_by(CC_2r) %>%
    mutate(
        across(c(cloud_cover, cloud_cover_DETER), ~ . %>% cummean_na(), .names = "{.col}_cum")
    ) %>%
    ungroup()

# ------------------------------------------------------------------------------
# Upstream Deforestation Aggregation
# ------------------------------------------------------------------------------

analysis_subset <- analysis %>%
    filter(year %in% 2004:2020, CC_2r %in% legal_amazon) %>%
    group_by(year) %>%
    mutate(
        upstream_weights = map(CC_2r, extraction_worker),
        across(forest:mining, \(x) weighted_sum_worker(x, upstream_weights, CC_2r), .names = "{.col}_upstream"),
        across(c(cloud_cover, cloud_cover_DETER), \(x) weighted_sum_worker(x, upstream_weights, CC_2r), .names = "{.col}_upstream"),
        total_upstream = weighted_sum_worker(total, upstream_weights, CC_2r)
    ) %>%
    ungroup() %>%
    select(-upstream_weights) %>%
    mutate(
        across(forest_upstream:mining_upstream, ~ . / total_upstream, .names = "{.col}_share")
    ) %>%
    group_by(CC_2r) %>%
    mutate(
        across(forest_upstream_share:mining_upstream_share, ~ . - lag(., 1), .names = "{.col}_d")
    ) %>%
    ungroup() %>%
    filter(year != 2004)

# ==============================================================================
# BALANCED PANEL CREATION
# ==============================================================================

# ------------------------------------------------------------------------------
# Small Panel (2010-2017) - Total Mortality
# ------------------------------------------------------------------------------

variable_names <- c("mortality_rate_tot", "hosp_rate", "forest_upstream_share_d", "cloud_cover_DETER_upstream", "temperature", "precipitation", "gdp_pc", "educ_ideb", "vaccination_index_5y", "health_primary_care_coverage", "health_doctors_1000")

small_panel <- analysis_subset %>%
    filter(year %in% 2010:2017) %>%
    filter(CC_2r %in% get_full_panel_muns(., variable_names)) %>%
    rename(municipality = CC_2r) %>%
    select(
        municipality, year, region, region_year,
        mortality_rate_tot, mortality_rate_l5, hosp_rate, ex_pop,
        low_birth_weight, low_birth_weight_share,               # <-- added
        forest_upstream_share:mining_upstream_share,
        forest_upstream_share_d:mining_upstream_share_d,
        cloud_cover_DETER_upstream,
        temperature, precipitation, gdp_pc, educ_ideb, vaccination_index_5y,
        health_primary_care_coverage, health_doctors_1000
    )

small_panel %>% check_balance(municipality, year)
small_panel %>% write_csv("data/analysis/small_panel.csv")

# ------------------------------------------------------------------------------
# Small Panel (2010-2017) - Infant Mortality
# ------------------------------------------------------------------------------

variable_names <- c("mortality_rate_l1", "hosp_rate", "forest_upstream_share_d", "cloud_cover_DETER_upstream", "temperature", "precipitation", "gdp_pc", "educ_ideb", "vaccination_index_5y", "health_primary_care_coverage", "health_doctors_1000")

small_panel_l1 <- analysis_subset %>%
    filter(year %in% 2010:2017) %>%
    filter(CC_2r %in% get_full_panel_muns(., variable_names)) %>%
    rename(municipality = CC_2r) %>%
    select(
        municipality, year, region, region_year,
        mortality_rate_l1, hosp_rate, ex_pop,
        low_birth_weight, low_birth_weight_share,               # <-- added
        forest_upstream_share:mining_upstream_share,
        forest_upstream_share_d:mining_upstream_share_d,
        cloud_cover_DETER_upstream,
        temperature, precipitation, gdp_pc, educ_ideb, vaccination_index_5y,
        health_primary_care_coverage, health_doctors_1000
    )

small_panel_l1 %>% check_balance(municipality, year)
small_panel_l1 %>% write_csv("data/analysis/small_panel_l1.csv")

# ------------------------------------------------------------------------------
# Large Panel (2005-2017) - Total Mortality
# ------------------------------------------------------------------------------

variable_names <- c("mortality_rate_tot", "hosp_rate", "forest_upstream_share_d", "cloud_cover_DETER_upstream", "temperature", "precipitation", "gdp_pc", "educ_ideb", "vaccination_index_5y")

large_panel <- analysis_subset %>%
    filter(year %in% 2005:2017) %>%
    filter(CC_2r %in% get_full_panel_muns(., variable_names)) %>%
    rename(municipality = CC_2r) %>%
    select(
        municipality, year, region, region_year,
        mortality_rate_tot, mortality_rate_l5, hosp_rate, ex_pop,
        low_birth_weight, low_birth_weight_share,               # <-- added
        forest_upstream_share:mining_upstream_share,
        forest_upstream_share_d:mining_upstream_share_d,
        cloud_cover_DETER_upstream,
        temperature, precipitation, gdp_pc, educ_ideb, vaccination_index_5y
    )

large_panel %>% check_balance(municipality, year)
large_panel %>% write_csv("data/analysis/large_panel.csv")

# ------------------------------------------------------------------------------
# Large Panel (2005-2017) - Infant Mortality
# ------------------------------------------------------------------------------

variable_names <- c("mortality_rate_l1", "hosp_rate", "forest_upstream_share_d", "cloud_cover_DETER_upstream", "temperature", "precipitation", "gdp_pc", "educ_ideb", "vaccination_index_5y")

large_panel_l1 <- analysis_subset %>%
    filter(year %in% 2005:2017) %>%
    filter(CC_2r %in% get_full_panel_muns(., variable_names)) %>%
    rename(municipality = CC_2r) %>%
    select(
        municipality, year, region, region_year,
        mortality_rate_l1, hosp_rate, ex_pop,
        low_birth_weight, low_birth_weight_share,               # <-- added
        forest_upstream_share:mining_upstream_share,
        forest_upstream_share_d:mining_upstream_share_d,
        cloud_cover_DETER_upstream,
        temperature, precipitation, gdp_pc, educ_ideb, vaccination_index_5y
    )

large_panel_l1 %>% check_balance(municipality, year)
large_panel_l1 %>% write_csv("data/analysis/large_panel_l1.csv")

# ------------------------------------------------------------------------------
# Census Panel
# ------------------------------------------------------------------------------

census <- read_parquet("data/misc/census.parquet")

census_agg <- census %>%
    filter(municipality > 1000000) %>%
    arrange(municipality, year) %>%
    group_by(CC_2r = str_sub(municipality, 1, 6), year) %>%
    summarise(
        pop_white = weighted.mean(race == "white", weight, na.rm = T),
        pop_from_municipality = weighted.mean(from_municipality != "no", weight, na.rm = T),
        pop_from_state = weighted.mean(from_state != "no", weight, na.rm = T),
    ) %>%
    ungroup()

rm(census)

variable_names <- c("mortality_rate_tot", "mortality_rate_l1", "hosp_rate", "forest", "temperature", "precipitation", "gdp_pc", "vaccination_index_5y")

census_panel <- inner_join(analysis, census_agg, by = c("CC_2r", "year")) %>%
    filter(CC_2r %in% legal_amazon) %>%
    group_by(year) %>%
    mutate(
        upstream_weights = map(CC_2r, extraction_worker),
        across(forest:mining, \(x) weighted_sum_worker(x, upstream_weights, CC_2r), .names = "{.col}_upstream"),
        across(c(cloud_cover, cloud_cover_DETER), \(x) weighted_sum_worker(x, upstream_weights, CC_2r), .names = "{.col}_upstream"),
        total_upstream = weighted_sum_worker(total, upstream_weights, CC_2r)
    ) %>%
    ungroup() %>%
    filter(CC_2r %in% get_full_panel_muns(., variable_names)) %>%
    rename(municipality = CC_2r) %>%
    select(-upstream_weights) %>%
    mutate(
        across(forest_upstream:mining_upstream, ~ . / total_upstream, .names = "{.col}_share"),
    ) %>%
    select(
        municipality, year, region_year,
        mortality_rate_tot, mortality_rate_l1, hosp_rate, ex_pop,
        low_birth_weight, low_birth_weight_share,               # <-- added
        forest_upstream_share:mining_upstream_share,
        precipitation, temperature, gdp_pc, vaccination_index_5y,
        pop_white, pop_from_municipality
    )

check_balance(census_panel, municipality, year)
census_panel %>% write_csv("data/analysis/census_panel.csv")

# ------------------------------------------------------------------------------
# Placebo Panel
# ------------------------------------------------------------------------------

placebo_panel <- analysis %>%
    drop_na(forest_d) %>%
    mutate(
        forest_d_share = forest_d / total,
        DETER_active = year >= 2005,
        legal_amazon = CC_2r %in% legal_amazon,
    ) %>%
    select(municipality = CC_2r, year, forest_d_share, DETER_active, legal_amazon, cloud_cover, cloud_cover_DETER)

placebo_panel %>% write_csv("data/analysis/placebo_panel.csv")

# ==============================================================================
# SPATIAL WEIGHTING MATRIX
# ==============================================================================

municipalities_simplified <- municipalities_simplified %>% st_transform(5641)

centroids <- st_centroid(municipalities_simplified)
coords <- st_coordinates(centroids)
distance_matrix <- as.matrix(dist(coords))

inv_dist_weights <- ifelse((distance_matrix == 0 | distance_matrix > 500 * 1e3), 0, 1 / distance_matrix)
inv_dist_weights <- Matrix(inv_dist_weights, sparse = TRUE)
inv_dist_weights <- inv_dist_weights / rowSums(inv_dist_weights)
inv_dist_weights <- round(inv_dist_weights, 4)
inv_dist_weights <- as(inv_dist_weights, "dgCMatrix")

writeMM(inv_dist_weights, "data/misc/municipality_weights.mtx")
municipalities_simplified %>%
    st_drop_geometry() %>%
    select(CC_2r) %>%
    write_csv("data/misc/municipality_weights.csv")

# ==============================================================================
# VISUALIZATIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# Variable Temporal Ranges
# ------------------------------------------------------------------------------

all_variables <- c("mortality_rate_tot", "mortality_rate_l1", "hosp_rate", "ex_pop", "deforestation", "cloud_cover", "cloud_cover_DETER", "temperature", "precipitation", "gdp_pc", "educ_ideb", "vaccination_index_5y", "health_primary_care_coverage", "health_doctors_1000")
all_variables_labels <- c("Mortality share (total)", "Mortality share (under 1)", "Hospitalization share", "Expenditure per capita", "Deforestation share", "Cloud cover", "Cloud cover (DETER)", "Temperature", "Precipitation", "GDP per capita", "Education (IDEB)", "Vaccination index (5y)", "Primary care coverage", "Doctors per 1000")

analysis %>%
    summarise(across(c(all_of(all_variables)), ~ list(range(year[!is.na(.)] %>% as.numeric())))) %>%
    pivot_longer(cols = everything(), names_to = "variable", values_to = "range") %>%
    mutate(
        ymin = map_dbl(range, ~ min(.)),
        ymax = map_dbl(range, ~ max(.))
    ) %>%
    ggplot(aes(x = ymin, xend = ymax, y = factor(variable), yend = factor(variable))) +
    geom_segment(linewidth = 5, lineend = "round") +
    scale_y_discrete(limits = rev(all_variables), labels = rev(all_variables_labels)) +
    labs(x = "Year", y = NULL) +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 12), axis.text.x = element_text(size = 12), axis.title = element_text(size = 14))

ggsave("output/figures/variable_temporal_ranges.png", width = 7, height = 5, dpi = 300)

# ------------------------------------------------------------------------------
# Panel Maps
# ------------------------------------------------------------------------------

legal_amazon_boundary <- municipalities_simplified %>%
    filter(CC_2r %in% legal_amazon) %>%
    st_union() %>%
    st_boundary()
legal_amazon_boundary <- st_cast(legal_amazon_boundary, "MULTILINESTRING") %>% st_cast("LINESTRING")
legal_amazon_boundary <- legal_amazon_boundary[st_length(legal_amazon_boundary) == max(st_length(legal_amazon_boundary))]

large_panel_plot <- ggplot() +
    geom_sf(data = municipalities_simplified, fill = "grey90", color = "black") +
    geom_sf(data = municipalities_simplified %>% filter(CC_2r %in% large_panel$municipality), aes(fill = "In Sample"), color = "black") +
    geom_sf(data = legal_amazon_boundary, fill = NA, aes(color = "Legal Amazon"), linewidth = 2) +
    scale_color_manual(values = c("Legal Amazon" = "purple"), name = "") +
    scale_fill_manual(values = c("In Sample" = "yellow"), name = "") +
    theme_minimal() +
    theme(legend.position = "inside", legend.position.inside = c(.9, .3))

ggsave("output/figures/large_panel_map.png", plot = large_panel_plot, width = 7, height = 7, dpi = 300)

small_panel_plot <- ggplot() +
    geom_sf(data = municipalities_simplified, fill = "grey90", color = "black") +
    geom_sf(data = municipalities_simplified %>% filter(CC_2r %in% small_panel$municipality), aes(fill = "In Sample"), color = "black") +
    geom_sf(data = legal_amazon_boundary, fill = NA, aes(color = "Legal Amazon"), linewidth = 2) +
    scale_color_manual(values = c("Legal Amazon" = "purple"), name = "") +
    scale_fill_manual(values = c("In Sample" = "yellow"), name = "") +
    theme_minimal() +
    theme(legend.position = "inside", legend.position.inside = c(.9, .3))

ggsave("output/figures/small_panel_map.png", plot = small_panel_plot, width = 7, height = 7, dpi = 300)

# ------------------------------------------------------------------------------
# Weights Example
# ------------------------------------------------------------------------------

weights_viz <- Matrix::readMM("data/river_network/processed/weights_matrix_exponential_1000.mtx")
weights_municipalities_viz <- read_csv("data/river_network/processed/weights_municipalities.csv")
rownames(weights_viz) <- weights_municipalities_viz$cc_2r
colnames(weights_viz) <- weights_municipalities_viz$cc_2r

max_muns <- list(
    names(which.max(rowSums(weights_viz[str_sub(rownames(weights_viz), 1, 1) == "3", ] > 0))),
    names(which.max(rowSums(weights_viz[str_sub(rownames(weights_viz), 1, 1) == "1", ] > 0)))
)

weights_plot <- ggplot() +
    geom_sf(data = municipalities_simplified, fill = "grey", color = "black", linewidth = .01) +
    geom_sf(data = left_join(municipalities_simplified, tibble(weight = weights_viz[max_muns[[1]], ], CC_2r = rownames(weights_viz))) %>% filter(weight > 0), aes(fill = weight), color = "black", linewidth = .01) +
    geom_sf(data = left_join(municipalities_simplified, tibble(weight = weights_viz[max_muns[[2]], ], CC_2r = rownames(weights_viz))) %>% filter(weight > 0), aes(fill = weight), color = "black", linewidth = .01) +
    geom_sf(data = municipalities_simplified %>% filter(CC_2r %in% max_muns), fill = "black", color = "black") +
    scale_fill_viridis_c(trans = "log10") +
    theme_minimal() +
    theme(legend.position = "inside", legend.position.inside = c(.9, .3))

ggsave("output/figures/weights_example.png", plot = weights_plot, width = 6, height = 6, dpi = 300)

# ==============================================================================
# TABLES
# ==============================================================================

# ------------------------------------------------------------------------------
# Mapbiomas Legend Table
# ------------------------------------------------------------------------------

legend <- readxl::read_excel("data/land_cover/mapbiomas_legend.xlsx")

latex_table <- legend %>%
    mutate(
        "Mapbiomas Class ID" = str_extract(Class, "^(\\d+\\.)*"),
        "Mapbiomas Class Name" = str_extract(Class, "([A-Z][a-z]+\\s?)+"),
        "Own Class" = case_when(
            grepl("(^1\\.)", Class) ~ "Forest",
            grepl("(^3\\.1)", Class) ~ "Pasture",
            grepl("(^3\\.2)", Class) ~ "Agriculture",
            grepl("(^4\\.2)", Class) ~ "Urban",
            grepl("(^4\\.3)", Class) ~ "Mining",
            TRUE ~ "Other"
        )
    ) %>%
    select(`Mapbiomas Class ID`:`Own Class`) %>%
    kableExtra::kable(
        format = "latex", booktabs = TRUE, linesep = "",
        caption = "Mapbiomas ID Equivalence"
    ) %>%
    as.character()

lines <- strsplit(latex_table, "\n")[[1]]
lines <- append(lines, "\\label{tbl-mapbiomas-equivalence}", after = 3)
latex_table_with_label <- paste(lines, collapse = "\n")

writeLines(latex_table_with_label, "output/tables/mapbiomas_equivalence.tex")

# ==============================================================================
# DESCRIPTIVE STATISTICS
# ==============================================================================

# ------------------------------------------------------------------------------
# Land Cover Changes
# ------------------------------------------------------------------------------

# Brazil-wide forest cover
analysis %>%
    group_by(year) %>%
    summarise(forest = sum(sum(forest, na.rm = T)) / sum(total, na.rm = T)) %>%
    filter(year %in% c(1986, 2020))
# year forest: 1986 = 0.685, 2020 = 0.586

# Legal Amazon land cover
analysis %>%
    filter(CC_2r %in% legal_amazon) %>%
    group_by(year) %>%
    summarise(
        forest = sum(sum(forest, na.rm = T)) / sum(total, na.rm = T),
        pasture = sum(sum(pasture, na.rm = T)) / sum(total, na.rm = T),
        agriculture = sum(sum(agriculture, na.rm = T)) / sum(total, na.rm = T),
        urban = sum(sum(urban, na.rm = T)) / sum(total, na.rm = T),
        mining = sum(sum(mining, na.rm = T)) / sum(total, na.rm = T)
    ) %>%
    filter(year %in% c(1986, 2020))
# 1986: forest=0.845, pasture=0.0449, agriculture=0.00690, urban=0.000454, mining=0.0000698
# 2020: forest=0.726, pasture=0.145, agriculture=0.0399, urban=0.00120, mining=0.000492

# ------------------------------------------------------------------------------
# Topology Statistics
# ------------------------------------------------------------------------------

topology <- arrow::read_feather("/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis/data/river_network/topology.feather")

# topology$estuary %>% sum()      # 1025
# topology$confluence %>% sum()   # 123140
# topology$source %>% sum()       # 125192

# ------------------------------------------------------------------------------
# Health Outcomes Summary (Work in Progress)
# ------------------------------------------------------------------------------

health_outcomes_summary <- tibble(
    variable = c("mortality_rate_tot", "mortality_rate_l1", "hosp_rate", "ex_pop"),
    variable_label = c("Mortality Rate (Total)", "Infant Mortality Rate", "Hospitalization Rate", "Expenditure per Capita")
)

map2(
    health_outcomes_summary %>% pull("variable"),
    health_outcomes_summary %>% pull("variable"),
    ~ list(
        variable = .y,
        min = min(analysis[[.x]], na.rm = TRUE),
        mean = mean(analysis[[.x]], na.rm = TRUE),
        median = median(analysis[[.x]], na.rm = TRUE),
        max = max(analysis[[.x]], na.rm = TRUE)
    )
) %>%
    bind_rows()

# ==============================================================================
# EXPLORATORY ANALYSIS (Development)
# ==============================================================================

# Census data exploration
census_raw <- read_csv("data/misc/raw/census_2010_2022.csv") %>%
    reframe(
        CC_2r = id_municipio %>% as.character() %>% str_sub(1, 6),
        year = ano,
        female = map_lgl(sexo, ~ . == "Mulheres"),
        race = cor_raca %>%
            factor(levels = c("Amarela", "Branca", "Indígena", "Preta", "Parda")) %>%
            forcats::fct_recode(
                "Asian" = "Amarela",
                "White" = "Branca",
                "Indigenous" = "Indígena",
                "Black" = "Preta",
                "Brown" = "Parda"
            ),
        population = populacao_residente
    )

census_agg_raw <- census_raw %>%
    group_by(CC_2r, year) %>%
    summarise(
        pop_total = sum(population, na.rm = T),
        pop_white = sum(population[race == "White"], na.rm = T) / pop_total
    ) %>%
    ungroup()

internal_change <- full_join(mortality_yy, births, by = c("CC_2r", "year")) %>%
    arrange(CC_2r, year) %>%
    group_by(CC_2r) %>%
    reframe(
        yyear = year[year >= 2010],
        net_internal_change_r_2010 = cumsum(deaths[year >= 2010]) - cumsum(total_births[year >= 2010])
    ) %>%
    fill(net_internal_change_r_2010) %>%
    ungroup() %>%
    rename(year = yyear)



