library(tidyverse)
library(sf)
library(arrow)
library(sfarrow)
library(lfe)
library(stargazer)

# read data
water_quality <- read_parquet("data/water_quality/quality_indicators_panel.parquet")

land_cover <- read_parquet("data/land_cover/land_cover_stations.parquet")

stations <- st_read_feather("data/water_quality/stations_rivers.feather") %>%
    st_set_crs(5641)

boundaries <- st_read("data/boundaries/gadm41_BRA_0.json") %>% st_transform(5641)

# merge data
analysis <- left_join(water_quality, land_cover, by = c("station", "year"))

###
# Helper Functions
###

dep_vars = c("pH", "turbidity", "biochem_oxygen_demand", "dissolved_oxygen", "total_residue", "total_nitrogen", "nitrates")

extract_results <- function(x, outcome = NULL) {
    tmp <- x$model %>% map_df(~ .x %>% broom::tidy())
    if (is.null(outcome)) {
        return(tmp %>% select(coefficient = estimate, s_d = std.error, p_value = p.value))
    } else {
        return(tmp %>% filter(term == outcome) %>% select(coefficient = estimate, s_d = std.error, p_value = p.value))
    }
}

plot_results <- function(x, limits = NULL, multiple_ind_var = NULL) {
    if (is.null(limits)) {
        limits <- levels(x$bins)
    }
    if (!is.null(multiple_ind_var)) {
        ggplot(aes(x = bins, y = coefficient), data = x) +
            geom_point() +
            geom_errorbar(aes(ymin = coefficient - 1.96 * s_d, ymax = coefficient + 1.96 * s_d)) +
            geom_hline(yintercept = 0, linetype = "dashed") +
            ggh4x::facet_grid2(vars(dep_var), vars(ind_var), scales = "free_y", independent = "y") +
            scale_x_discrete(
                labels = str_extract_all(limits, "\\d+") %>% map_chr(~ paste0(as.double(.x[[1]]) / 1e3, "-", as.double(.x[[3]]) / 1e3, "km")),
                limits = limits
            ) +
            theme_minimal() +
            theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
            labs(x = "Upstream Query Distance", y = "Coefficient")
    } else {
        return(
            ggplot(aes(x = bins, y = coefficient), data = x) +
                geom_point() +
                geom_errorbar(aes(ymin = coefficient - 1.96 * s_d, ymax = coefficient + 1.96 * s_d)) +
                geom_hline(yintercept = 0, linetype = "dashed") +
                facet_grid(rows = vars(dep_var), scales = "free_y") +
                scale_x_discrete(
                    labels = str_extract_all(limits, "\\d+") %>% map_chr(~ paste0(as.double(.x[[1]]) / 1e3, "-", as.double(.x[[3]]) / 1e3, "km")),
                    limits = limits
                ) +
                theme_minimal() +
                theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
                labs(x = "Upstream Query Distance", y = "Coefficient")
        )
    }
}

###
# Descriptive Analysis
###

(analysis$distance_from_estuary / 1000) %>% summary()

water_quality %>%
    pull(year) %>%
    summary()

# plot number of observations by distance
analysis %>%
    filter(total != 0) %>%
    group_by(bins) %>%
    summarise(n = n()) %>%
    ggplot(aes(x = bins, y = n)) +
    geom_bar(stat = "identity") +
    scale_x_discrete(
        labels = str_extract_all(levels(analysis$bins), "\\d+") %>% map_chr(~ paste0(as.double(.x[[1]]) / 1e3, "-", as.double(.x[[3]]) / 1e3, "km")),
        limits = levels(analysis$bins)
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    labs(x = "Upstream Query Distance", y = "Number of Observations")

# plot number of observations by year
analysis %>%
    filter(total != 0) %>%
    group_by(year) %>%
    summarise(n = n()) %>%
    ggplot(aes(x = year, y = n)) +
    geom_bar(stat = "identity") +
    theme_minimal() +
    labs(x = "Year", y = "Number of Observations")

# plot all stations and their numbers of observations
stations_descriptive <- water_quality %>%
    drop_na(pH) %>%
    group_by(station) %>%
    summarise(n = n()) %>%
    left_join(stations %>% select(station = Codigo, geometry), by = "station")

ggplot() +
    geom_sf(data = stations_descriptive, aes(size = n, geometry = geometry), alpha = .5) +
    geom_sf(data = boundaries, fill = "transparent", color = "black", size = .3) +
    scale_size_continuous(range = c(.1, 2), guide = "none") +
    coord_sf(crs = 5641) +
    theme_void()

ggsave("output/figures/water_quality_stations_observations.png", width = 5, height = 5, bg = "white")

# get variance after demeaning
analysis %>%
    drop_na(deforestation) %>%
    filter(total != 0) %>%
    pull(deforestation) %>%
    sd()

# get variance after demeaning
analysis %>%
    drop_na(deforestation) %>%
    filter(total != 0) %>%
    group_by(station) %>%
    mutate(deforestation = deforestation - mean(deforestation)) %>%
    ungroup() %>%
    pull(deforestation) %>%
    sd()

analysis %>%
    drop_na(deforestation) %>%
    filter(total != 0) %>%
    group_by(station) %>%
    mutate(deforestation = deforestation - mean(deforestation)) %>%
    ungroup() %>%
    group_by(year) %>%
    mutate(deforestation = deforestation - mean(deforestation)) %>%
    ungroup() %>%
    pull(deforestation) %>%
    sd()

###
# General Analysis
###

analysis_single <- analysis %>%
    arrange(station, year, bins) %>%
    group_by(station, bins) %>%
    mutate(across(deforestation:deforestation_m, cumsum)) %>%
    ungroup() %>%
    group_by(station, year) %>%
    summarise(across(pH:nitrates, unique), across(deforestation:total, sum)) %>%
    ungroup() %>%
    filter(total != 0, pH != 0) %>%
    mutate(
        forest_share = forest / total,
        deforestation_share = deforestation / total
    )

analysis_single %>%
    group_by(station) %>%
    filter(sum(is.na(turbidity)) < .1) %>%
    mutate(across(turbidity, zoo::na.approx)) %>%
    ungroup()


ggplot(analysis_single) +
    geom_point(aes(x = deforestation_share, y = pH), alpha = .3) +
    theme_bw()

ggplot(analysis_single) +
    geom_boxplot(aes(group = as.factor(year), y = pH / 13), alpha = .3) +
    #geom_smooth(aes(x = year, y = pH / 13), method = "lm", se = FALSE) +
    theme_bw()

analysis %>%
    group_by(year) %>%
    summarise(across(total, sum)) %>%
    ggplot() +
    geom_line(aes(x = year, y = total))

ggplot(analysis_single) +
    geom_boxplot(aes(group = as.factor(year), y = deforestation_share), alpha = .3) +
    #geom_smooth(aes(x = year, y = pH / 13), method = "lm", se = FALSE) +
    theme_bw()

# prepare analysis grid
analysis_results <- tibble(
    dep_var = dep_vars
)

# function to filter and impute missing values for given variable
filter_impute <- function(x, variable){
    variable_sym <- sym(variable)
    
    x %>%
        filter(year >= 2005) %>%
        group_by(station) %>%
        filter(sum(is.na(!!variable_sym)) / n() < 1) %>%
        mutate(across(all_of(variable), ~ zoo::na.approx(., na.rm = FALSE))) %>%
        ungroup()
}


analysis_results <- analysis_results %>%
    mutate(formula = map(dep_var, ~ as.formula(paste0(.x, " ~ deforestation_share | station + year | 0 | station + year"))),
           model = map2(formula, dep_var, ~ felm(.x, data = filter_impute(analysis_single, .y))))

# hacky way of forcing stargazer to not obtain the formula from call
analysis_results$model[[1]]$call <- NULL

stargazer(analysis_results$model, type = "text") #, dep.var.labels = analysis_results$dep_var


###
# Default-model Analysis
###

analysis_single <- analysis %>%
    arrange(station, year, bins) %>%
    group_by(station, bins) %>%
    mutate(across(deforestation:deforestation_m, cumsum)) %>%
    ungroup() %>%
    filter(total != 0)

# prepare analysis grid
analysis_grid <- expand_grid(
    dep_var = c("pH", "turbidity", "biochem_oxygen_demand", "dissolved_oxygen", "total_residue", "total_nitrogen", "nitrates"),
    bins = levels(analysis_single$bins)
)

# ## EXPERIMENTAL
# ## Single Model

# analysis_single <- analysis_single %>%
#     pivot_wider(
#         id_cols = c(station, year, nitrates),
#         names_from = bins,
#         names_prefix = "deforestation_",
#         values_from = deforestation
#     ) %>%
#     mutate(across(deforestation_20:deforestation_200, function(x) log(x + .001)))

# felm(as.formula(paste0("nitrates ~ ", paste("deforestation_", seq(20, 200, 20), sep = "", collapse = " + "), " | station  | 0 | station")), analysis_single) %>% summary()

## Deforestation; No FE; No Cluster; Log-Log

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0("log(", .x, " + 1) ~ log(deforestation + 1)")), data = analysis_single %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results, "log(deforestation + 1)")

analysis_results %>% plot_results(limits = levels(analysis_single$bins))

ggsave("output/figures/stations_deforestation_d_log_Nfe_Ncl.png", width = 5, height = 10, bg = "white")

## Deforestation; One-Way FE; No Cluster; Log-Log

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0("log(", .x, " + 1) ~ log(deforestation + 1) | station")), data = analysis_single %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results, "log(deforestation + 1)")

analysis_results %>% plot_results(limits = levels(analysis_single$bins))

ggsave("output/figures/stations_deforestation_d_log_Ofe_Ncl.png", width = 5, height = 10, bg = "white")

## Deforestation; One-Way FE; Cluster; Log-Log

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0("log(", .x, " + 1) ~ log(deforestation + 1) | station | 0 | station")), data = analysis_single %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results)

analysis_results %>% plot_results(limits = levels(analysis_single$bins))

ggsave("output/figures/stations_deforestation_d_log_Ofe_Ycl.png", width = 5, height = 10, bg = "white")

###
# Deforestation Share
###

analysis_single <- analysis %>%
    arrange(station, year, bins) %>%
    group_by(station, bins) %>%
    mutate(across(deforestation:deforestation_m, cumsum)) %>%
    ungroup() %>%
    mutate(
        forest_share = forest / total,
        deforestation_share = deforestation / total,
        deforestation_f_share = deforestation_f / total,
        deforestation_p_share = deforestation_p / total,
        deforestation_u_share = deforestation_u / total,
        deforestation_m_share = deforestation_m / total,
    ) %>%
    filter(total != 0)

# prepare analysis grid
analysis_grid <- expand_grid(
    dep_var = c("pH", "turbidity", "biochem_oxygen_demand", "dissolved_oxygen", "total_residue", "total_nitrogen", "nitrates"),
    bins = levels(analysis_single$bins)
)

## Deforestation Share; No FE; No Cluster

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0(.x, " ~ deforestation_share")), data = analysis_single %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results, "deforestation_share")

analysis_results %>% plot_results(limits = levels(analysis_single$bins))

ggsave("output/figures/stations_deforestation_d_share_Nfe_Ncl.png", width = 5, height = 10, bg = "white")

## Deforestation Share; One-Way FE; No Cluster

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0(.x, " ~ deforestation_share | station")), data = analysis_single %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results, "deforestation_share")

analysis_results %>% plot_results(limits = levels(analysis_single$bins))

ggsave("output/figures/stations_deforestation_d_share_Ofe_Ncl.png", width = 5, height = 10, bg = "white")

## Deforestation Share; One-Way FE; Cluster

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0(.x, " ~ deforestation_share | station | 0 | station")), data = analysis_single %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results)

analysis_results %>% plot_results(limits = levels(analysis_single$bins))

ggsave("output/figures/stations_deforestation_d_share_Ofe_Ycl.png", width = 5, height = 10, bg = "white")

## Deforestation Share; FE; No Cluster

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0(.x, " ~ deforestation_share | station + year")), data = analysis_single %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results, "deforestation_share")

analysis_results %>% plot_results(limits = levels(analysis_single$bins))

ggsave("output/figures/stations_deforestation_d_share_Yfe_Ncl.png", width = 5, height = 10, bg = "white")

## Deforestation Share; FE; Cluster

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0(.x, " ~ deforestation_share | station + year | 0 | station + year")), data = analysis_single %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results)

analysis_results %>% plot_results(limits = levels(analysis_single$bins))

ggsave("output/figures/stations_deforestation_d_share_Yfe_Ycl.png", width = 5, height = 10, bg = "white")

## Deforestation Share - by replacement; FE; No Cluster

analysis_grid_replacement <- expand_grid(
    dep_var = c("pH", "turbidity", "biochem_oxygen_demand", "dissolved_oxygen", "total_residue", "total_nitrogen", "nitrates"),
    ind_var = c("deforestation_share", "deforestation_f_share", "deforestation_p_share", "deforestation_u_share", "deforestation_m_share"),
    bins = levels(analysis_single$bins),
)

analysis_results <- analysis_grid_replacement %>%
    mutate(model = pmap(list(dep_var, ind_var, bins), function(.dep_var, .ind_var, .bins) felm(as.formula(paste0(.dep_var, " ~ ", .ind_var, " | station + year")), data = analysis_single %>% filter(bins == .bins))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results)

analysis_results %>% plot_results(limits = levels(analysis_single$bins), multiple_ind_var = "ind_var")

###
# Cumulative Analysis
###

# compute cumulative sums
analysis_cumulative <- analysis %>%
    arrange(station, year, bins) %>%
    group_by(station, year) %>%
    mutate(across(deforestation:total, cumsum)) %>%
    ungroup() %>%
    group_by(station, bins) %>%
    mutate(across(deforestation:deforestation_m, cumsum)) %>%
    ungroup()

# calculate additional variables
analysis_cumulative <- analysis_cumulative %>%
    mutate(
        forest_share = forest / total,
        deforestation_share = deforestation / total
    )

# analysis_cumulative = analysis_cumulative %>%
#     mutate(bins = ordered(bins, levels = levels(analysis_cumulative$bins)))
levels(analysis_cumulative$bins) <- c("[0.0,20000.0)", "[0.0,40000.0)", "[0.0,60000.0)", "[0.0,80000.0)", "[0.0,100000.0)", "[0.0,120000.0)", "[0.0,140000.0)", "[0.0,160000.0)", "[0.0,180000.0)", "[0.0,200000.0)")


# prepare analysis grid
analysis_grid <- expand_grid(
    dep_var = c("pH", "turbidity", "biochem_oxygen_demand", "dissolved_oxygen", "total_residue", "total_nitrogen", "nitrates"),
    bins = levels(analysis_cumulative$bins)
)

## Deforestation; No FE; No Cluster

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0(.x, " ~ deforestation")), data = analysis_cumulative %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results, "deforestation")

analysis_results %>% plot_results(limits = c("[0.0,20000.0)", "[0.0,40000.0)", "[0.0,60000.0)", "[0.0,80000.0)", "[0.0,100000.0)", "[0.0,120000.0)", "[0.0,140000.0)", "[0.0,160000.0)", "[0.0,180000.0)", "[0.0,200000.0)"))

ggsave("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/output/figures/stations_deforestation_Nfe_Ncl.png", width = 5, height = 10, bg = "white")

## Deforestation; One-Way FE; No Cluster

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0(.x, " ~ deforestation | station")), data = analysis_cumulative %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results, "deforestation")

analysis_results %>% plot_results(limits = c("[0.0,20000.0)", "[0.0,40000.0)", "[0.0,60000.0)", "[0.0,80000.0)", "[0.0,100000.0)", "[0.0,120000.0)", "[0.0,140000.0)", "[0.0,160000.0)", "[0.0,180000.0)", "[0.0,200000.0)"))

ggsave("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/output/figures/stations_deforestation_Ofe_Ncl.png", width = 5, height = 10, bg = "white")

## Deforestation; One-Way FE; Cluster

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0("log(", .x, " + .001) ~ log(deforestation + .001) | station | 0 | station")), data = analysis_cumulative %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results)

analysis_results %>% plot_results(limits = c("[0.0,20000.0)", "[0.0,40000.0)", "[0.0,60000.0)", "[0.0,80000.0)", "[0.0,100000.0)", "[0.0,120000.0)", "[0.0,140000.0)", "[0.0,160000.0)", "[0.0,180000.0)", "[0.0,200000.0)"))

ggsave("output/figures/stations_deforestation_Ofe_Ycl.png", width = 5, height = 10, bg = "white")

### TODO:
# show limited variance after two-way clustering
# restrict to rural sample; no urbanizations upstream

## Deforestation; FE; No Cluster

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0(.x, " ~ deforestation | station + year")), data = analysis_cumulative %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results, "deforestation")

analysis_results %>% plot_results(limits = c("[0.0,20000.0)", "[0.0,40000.0)", "[0.0,60000.0)", "[0.0,80000.0)", "[0.0,100000.0)", "[0.0,120000.0)", "[0.0,140000.0)", "[0.0,160000.0)", "[0.0,180000.0)", "[0.0,200000.0)"))

ggsave("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/output/figures/stations_deforestation_Yfe_Ncl.png", width = 5, height = 10, bg = "white")

## Deforestation; FE; Cluster

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0("log(", .x, " + 0.001) ~ log(deforestation + 0.001) | station + year | 0 | station + year")), data = analysis_cumulative %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results)

analysis_results %>% plot_results(limits = levels(analysis_cumulative$bins))

ggsave("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/output/figures/stations_deforestation_Yfe_Ycl.png", width = 5, height = 10, bg = "white")


## Deforestation - by replacement; FE; No Cluster

analysis_grid_replacement <- expand_grid(
    dep_var = c("pH", "turbidity", "biochem_oxygen_demand", "dissolved_oxygen", "total_residue", "total_nitrogen", "nitrates"),
    ind_var = c("deforestation", "deforestation_f", "deforestation_p", "deforestation_u", "deforestation_m"),
    bins = levels(analysis_cumulative$bins),
)

analysis_results <- analysis_grid_replacement %>%
    mutate(model = pmap(list(dep_var, ind_var, bins), function(.dep_var, .ind_var, .bins) felm(as.formula(paste0(.dep_var, " ~ ", .ind_var, " | station + year")), data = analysis_cumulative %>% filter(bins == .bins))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results)

analysis_results %>% plot_results(limits = c("[0.0,20000.0)", "[0.0,40000.0)", "[0.0,60000.0)", "[0.0,80000.0)", "[0.0,100000.0)", "[0.0,120000.0)", "[0.0,140000.0)", "[0.0,160000.0)", "[0.0,180000.0)", "[0.0,200000.0)"), multiple_ind_var = "ind_var")

ggsave("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/output/figures/stations_deforestation_replacement_Nfe_Ncl.png", width = 10, height = 20, bg = "white")

ggplot(aes(x = bins, y = coefficient), data = analysis_results) +
    geom_point() +
    geom_errorbar(aes(ymin = coefficient - 1.96 * s_d, ymax = coefficient + 1.96 * s_d)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    ggh4x::facet_grid2(vars(dep_var), vars(ind_var), scales = "free_y", independent = "y") +
    scale_x_discrete(
        labels = paste0(seq(20, 200, 20), "km"),
        limits = c("[0.0,20000.0)", "[0.0,40000.0)", "[0.0,60000.0)", "[0.0,80000.0)", "[0.0,100000.0)", "[0.0,120000.0)", "[0.0,140000.0)", "[0.0,160000.0)", "[0.0,180000.0)", "[0.0,200000.0)"),
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    labs(x = "Upstream Query Distance", y = "Coefficient")


## Forest Share; No FE; No Cluster

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0(.x, " ~ forest_share")), data = analysis_cumulative %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results, "forest_share")

analysis_results %>% plot_results(limits = c("[0.0,20000.0)", "[0.0,40000.0)", "[0.0,60000.0)", "[0.0,80000.0)", "[0.0,100000.0)", "[0.0,120000.0)", "[0.0,140000.0)", "[0.0,160000.0)", "[0.0,180000.0)", "[0.0,200000.0)"))

ggsave("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/output/figures/stations_forest_share_Nfe_Ncl.png", width = 5, height = 10, bg = "white")

## Forest Share; FE; No Cluster

analysis_results <- analysis_grid %>%
    mutate(model = map2(dep_var, bins, ~ felm(as.formula(paste0(.x, " ~ forest_share | station + year")), data = analysis_cumulative %>% filter(bins == .y))))

analysis_results[c("coefficient", "s_d", "p_value")] <- extract_results(analysis_results, "forest_share")

analysis_results %>% plot_results(limits = c("[0.0,20000.0)", "[0.0,40000.0)", "[0.0,60000.0)", "[0.0,80000.0)", "[0.0,100000.0)", "[0.0,120000.0)", "[0.0,140000.0)", "[0.0,160000.0)", "[0.0,180000.0)", "[0.0,200000.0)"))

ggsave("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/output/figures/stations_forest_share_Yfe_Ncl.png", width = 5, height = 10, bg = "white")



felm(pH ~ forest | station + year | 0 | station + year, data = analysis %>% filter(bins == "[0.0,20000.0)")) %>% broom::tidy(.)


---


extract_results <- function(x, outcome = NULL) {
    tmp <- x$model %>% map_df(~ .x %>% broom::tidy())
    if (is.null(outcome)) {
        return(tmp %>% select(coefficient = estimate, s_d = std.error, p_value = p.value))
    } else {
        return(tmp %>% filter(term == outcome) %>% select(coefficient = estimate, s_d = std.error, p_value = p.value))
    }
}

plot_results <- function(x, limits = NULL, multiple_ind_var = NULL, facet_labels = NULL) {
    if (is.null(limits)) {
        limits <- levels(x$bins)
    }
    if (is.null(facet_labels)) {
        labeller <- as_labeller(c(
            pH = "pH",
            turbidity = "Turbidity",
            biochem_oxygen_demand = "Biochemical\nOxygen Demand",
            dissolved_oxygen = "Dissolved Oxygen",
            total_residue = "Total Residue",
            total_nitrogen = "Total Nitrogen",
            nitrates = "Nitrates"
        ))
    } else {
        labeller <- as_labeller(facet_labels)
    }
    if (!is.null(multiple_ind_var)) {
        ggplot(aes(x = bins, y = coefficient), data = x) +
            geom_point() +
            geom_errorbar(aes(ymin = coefficient - 1.96 * s_d, ymax = coefficient + 1.96 * s_d)) +
            geom_hline(yintercept = 0, linetype = "dashed") +
            ggh4x::facet_grid2(vars(dep_var), vars(ind_var), scales = "free_y", independent = "y") +
            scale_x_discrete(
                labels = str_extract_all(limits, "\\d+") %>% map_chr(~ paste0(as.double(.x[[1]]) / 1e3, "-", as.double(.x[[3]]) / 1e3, "km")),
                limits = limits
            ) +
            theme_bw() +
            theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
            labs(x = "Upstream Query Distance", y = "Coefficient")
    } else {
        return(
            ggplot(aes(x = bins, y = coefficient), data = x) +
                geom_point() +
                geom_errorbar(aes(ymin = coefficient - 1.96 * s_d, ymax = coefficient + 1.96 * s_d)) +
                geom_hline(yintercept = 0, linetype = "dashed") +
                facet_grid(rows = vars(dep_var), scales = "free_y", labeller = labeller) +
                scale_x_discrete(
                    labels = str_extract_all(limits, "\\d+") %>% map_chr(~ paste0(as.double(.x[[1]]) / 1e3, "-", as.double(.x[[3]]) / 1e3, "km")),
                    limits = limits
                ) +
                theme_bw() +
                theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
                labs(x = "Upstream Query Distance", y = "Coefficient")
        )
    }
}