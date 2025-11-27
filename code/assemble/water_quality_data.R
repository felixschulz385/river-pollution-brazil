library(tidyverse)
library(sf)
library(arrow)
library(sfarrow)
library(kableExtra)

###
# Helper Functions
###

standardize <- function(x) {
    return((x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE))
}

dep_vars = c("pH", "turbidity", "biochem_oxygen_demand", "dissolved_oxygen", "total_residue", "total_nitrogen", "nitrates")
dep_vars_label = c("pH", "Turbidity", "Biochemical\nOxygen\nDemand", "Dissolved\nOxygen", "Total\nResidue", "Total\nNitrogen", "Nitrates")

# read data
water_quality <- read_parquet("data/water_quality/quality_indicators_panel.parquet") %>% filter(pH != 0)

land_cover <- read_parquet("data/land_cover/land_cover_stations.parquet")

stations <- st_read_feather("data/water_quality/processed/stations_rivers.feather") %>%
    st_set_crs(5641) %>% filter(!duplicated(Codigo))

boundaries <- st_read("data/misc/raw/gadm/gadm41_BRA_0.json") %>% st_transform(5641) %>% st_simplify(dTolerance = 0.01)

# merge data
analysis <- left_join(water_quality, land_cover, by = c("station", "year"))

# Calculate the annual-station averages
analysis_panel <- analysis %>%
    arrange(station, year, bins) %>%
    group_by(station, year) %>%
    summarise(across(pH:nitrates, unique), across(deforestation:total, sum)) %>%
    ungroup() %>%
    filter(total > 100) %>%
    mutate(
        # deforestation_share = deforestation / total,
        # deforestation_p_share = deforestation_p / total,
        # deforestation_a_share = deforestation_a / total,
        # deforestation_u_share = deforestation_u / total,
        # deforestation_m_share = deforestation_m / total,
        forest_share = forest / total,
        pasture_share = pasture / total,
        agriculture_share = agriculture / total,
        urban_share = urban / total,
        mining_share = mining / total,
    ) %>%
    group_by(station) %>% 
    mutate(
        d_forest_share = - (forest_share - lag(forest_share, 1)),
        d_pasture_share = (pasture_share - lag(pasture_share, 1)),
        d_agriculture_share = (agriculture_share - lag(agriculture_share, 1)),
        d_urban_share = (urban_share - lag(urban_share, 1)),
        d_mining_share = (mining_share - lag(mining_share, 1))
        ) %>% 
    ungroup() %>%
    left_join(stations %>% select(station = Codigo, estuary, adm2), by = "station") %>%
    mutate(estuary_year = paste0(estuary, "_", year)) %>%
    select(station:nitrates, forest:estuary, estuary_year) %>%
    mutate(across(all_of(dep_vars), standardize, .names = "{.col}_sd"), 
           across(d_forest_share:d_mining_share, standardize, .names = "{.col}_sd"))

analysis_panel %>% write_csv("data/analysis/stations_panel.csv")

# aggregate in bins
analysis_bins <- analysis %>%
    arrange(station, year, bins) %>%
    group_by(station, year, bins) %>%
    summarise(across(pH:nitrates, unique), across(deforestation:total, sum)) %>%
    ungroup() %>%
    filter(total != 0) %>%
    mutate(
        forest_share = forest / total,
        pasture_share = pasture / total,
        agriculture_share = agriculture / total,
        urban_share = urban / total,
        mining_share = mining / total,
    ) %>%
    group_by(station, bins) %>% 
    mutate(
        d_forest_share = - (forest_share - lag(forest_share, 1)),
        d_pasture_share = (pasture_share - lag(pasture_share, 1)),
        d_agriculture_share = (agriculture_share - lag(agriculture_share, 1)),
        d_urban_share = (urban_share - lag(urban_share, 1)),
        d_mining_share = (mining_share - lag(mining_share, 1))
        ) %>% 
    ungroup() %>%
    left_join(stations %>% select(station = Codigo, estuary), by = "station") %>%
    mutate(estuary_year = paste0(estuary, "_", year)) %>%
    select(station:nitrates, forest:estuary, estuary_year) %>%
    mutate(distance = str_extract(bins, "\\d+") %>% as.double() %>% "/"(1000))

analysis_bins %>% write_csv("data/analysis/stations_bins.csv")

###
# Descriptive statistics
###

# number of stations
water_quality %>% pull(station) %>% unique() %>% length()

## available sensors over time

stations_time_bars <- ggplot(analysis_scatter) +
    geom_bar(aes(x = year_bin, fill = region), position = "stack") +
    scale_fill_viridis_d() +
    theme_bw() +
    labs(x = "Years", y = "Number of Sensors", fill = "Region") +
    theme(legend.position = "bottom")

ggsave("output/figures/stations_time_bars.png", stations_time_bars, width = 5, height = 4, bg = "white")


stations_time_spatial <- ggplot(analysis_scatter) +
    geom_sf(data = boundaries) +
    geom_sf(aes(geometry = geometry), data = analysis_scatter, size = .5) +
    facet_wrap(~year_bin) +
    theme_linedraw()

ggsave("output/figures/stations_time_spatial.png", stations_time_spatial, width = 10, height = 10, bg = "white")



# Natural limits
limits <- tibble(
  variable = c("pH", "turbidity", "biochem_oxygen_demand", "dissolved_oxygen", "total_residue", "total_nitrogen", "nitrates"),
  min_val = c(0, 0, 0, 0, 0, 0, 0),
  max_val = c(14, 100, 50, 200, 500, 100, 10)
)

# Reshape data to long format and merge with limits to get natural limits for each variable
water_quality_long <- water_quality %>%
    pivot_longer(pH:nitrates, names_to = "variable", values_to = "value") %>%
    #left_join(limits, by = "variable") %>%
    mutate(variable = factor(variable, levels = c("pH", "turbidity", "biochem_oxygen_demand", "dissolved_oxygen", "total_residue", "total_nitrogen", "nitrates")))

# Provide summary table for each variable
# Schema: variable, mean, median, min, max, number of missing values (rounded to 2 decimal places)
# format to latex using kableExtra, booktabs, label and caption
water_quality_summary <- water_quality_long %>%
    group_by(variable) %>%
    summarise(min = min(value, na.rm = TRUE),
              mean = mean(value, na.rm = TRUE),
              median = median(value, na.rm = TRUE),
              max = max(value, na.rm = TRUE),
              missing = mean(is.na(value))) %>%
    mutate(across(mean:missing, ~ round(.x, 2))) %>%
    mutate(variable = c("pH", "Turbidity", "Biochemical Oxygen Demand", "Dissolved Oxygen", "Total Residue", "Total Nitrogen", "Nitrates")) %>%
    `colnames<-`(c("", "Min", "Mean", "Median", "Max", "Missing"))


# Generate the LaTeX table string
latex_table <- water_quality_summary %>%
  kbl("latex", booktabs = TRUE, caption = "Summary Statistics of Sensor Data", linesep = "") %>%
  as.character()

# Insert the \label command after the third line
lines <- strsplit(latex_table, "\n")[[1]]
lines <- append(lines, "\\label{tbl-summary-sensor-data}", after = 3)
latex_table_with_label <- paste(lines, collapse = "\n")

# Write the modified LaTeX table string to a file
output_file <- "output/tables/summary_sensor_data.tex"
writeLines(latex_table_with_label, output_file)

###
# Scatterplots
###

municipalities <- st_read("data/misc/raw/gadm/gadm41_BRA_2.json") %>% rownames_to_column("adm2") %>% st_drop_geometry()
regions <- read_csv("data/misc/raw/brazil_regions_states.csv")
adm2_regions <- left_join(municipalities, regions, by = c("NAME_1" = "state")) %>% tibble() %>% reframe(adm2 = as.double(adm2), region)

analysis_scatter <- analysis_panel %>%
    # create bins of 5 years
    mutate(year_bin = cut(year, breaks = seq(1985, 2025, 10))) %>%
    left_join(stations, by = c("station" = "Codigo")) %>%
    left_join(adm2_regions, by = "adm2") %>%
    select(station, year, year_bin, estuary = estuary.x, region, forest_share:mining_share, pH, turbidity, biochem_oxygen_demand, dissolved_oxygen, total_residue, total_nitrogen, nitrates, geometry)

analysis_scatter_long <- analysis_scatter %>%
    select(-geometry) %>%
    pivot_longer(c(pH, turbidity, biochem_oxygen_demand, dissolved_oxygen, total_residue, total_nitrogen, nitrates), names_to = "variable", values_to = "value") %>%
    mutate(variable = factor(variable, levels = dep_vars, labels = dep_vars_label)) %>%
    filter(value > 0) %>%
    group_by(year_bin, variable) %>%
    mutate(value = value - mean(value, na.rm = T)) %>%
    ungroup() %>%
    group_by(station, variable) %>%
    mutate(value = value - mean(value, na.rm = T)) %>%
    ungroup()

## over time

regression_coefficients <- expand_grid(
    year_bin = unique(analysis_scatter_long$year_bin),
    variable = unique(analysis_scatter_long$variable)
    ) %>%
    mutate(
        coef = map2_dbl(year_bin, variable, ~ lm(value ~ forest_share, data = analysis_scatter_long %>% filter(year_bin == .x, variable == .y)) %>% coefficients() %>% pluck(2)),
        coef = paste0("β = ", round(coef, digits = 2))
        )

time_plot <- ggplot(analysis_scatter_long) +
    geom_jitter(aes(x = forest_share, y = value, color = variable), alpha = .5) +
    geom_smooth(aes(x = forest_share, y = value), method = "lm", color="black", se = TRUE) +
    geom_hline(yintercept = 0, linetype = "dashed", alpha = .6) +
    geom_text(aes(label = coef, x = -Inf, y = -Inf, ), hjust = -.1, vjust = -14, data = regression_coefficients) +
    labs(x = "", y = "Change in Forest Cover") +
    scale_y_continuous(transform = "pseudo_log") +
    ggh4x::facet_grid2(cols = vars(year_bin), rows = vars(variable), scales = "free", independent = "all") +
    theme_bw() +
    theme(legend.position = "none")

ggsave("output/figures/pollution_time_plot.png", time_plot, width = 10, height = 1.4142 * 10, bg = "white")

## between regions

regression_coefficients <- expand_grid(
    region = unique(analysis_scatter_long$region),
    variable = unique(analysis_scatter_long$variable)
    ) %>%
    mutate(
        coef = map2_dbl(region, variable, ~ lm(value ~ d_forest_share, data = analysis_scatter_long %>% filter(region == .x, variable == .y)) %>% coefficients() %>% pluck(2)),
        coef = paste0("β = ", round(coef, digits = 2))
        )

time_plot <- ggplot(analysis_scatter_long) +
    geom_jitter(aes(x = d_forest_share, y = value, color = variable), alpha = .5) +
    geom_smooth(aes(x = d_forest_share, y = value), method = "lm", color="black", se = TRUE) +
    geom_hline(yintercept = 0, linetype = "dashed", alpha = .6) +
    geom_text(aes(label = coef, x = -Inf, y = -Inf, ), hjust = -.1, vjust = -14, data = regression_coefficients) +
    labs(x = "", y = "Change in Forest Cover") +
    scale_y_continuous(transform = "pseudo_log") +
    ggh4x::facet_grid2(cols = vars(region), rows = vars(variable), scales = "free", independent = "all") +
    theme_bw() +
    theme(legend.position = "none")

ggsave("output/figures/pollution_region_plot.png", time_plot, width = 10, height = 1.4142 * 10, bg = "white")
