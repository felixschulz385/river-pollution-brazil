library(tidyverse)
library(lfe)
library(texreg)
library(splm)
library(Matrix)
library(spdep)

setwd("/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis")

# c("Cloud Cover", "Temperature", "Precipitation", "GDP per Capita", "Vaccination Index", "Doctors per 1000", "Education", "Primary Care Coverage")

###########################################
#                                         #
#             SENSOR STATIONS             #
#                                         #
###########################################

###
# Load data
###

analysis_panel <- read_csv("data/analysis/stations_panel.csv")
analysis_bins <- read_csv("data/analysis/stations_bins.csv")

dep_vars <- c("pH", "turbidity", "biochem_oxygen_demand", "dissolved_oxygen", "total_residue", "total_nitrogen", "nitrates")
dep_vars_table <- c("pH", "Turbidity", "\\makecell{Biochemical\\\\Oxygen\\\\Demand}", "\\makecell{Dissolved\\\\Oxygen}", "\\makecell{Total\\\\Residue}", "\\makecell{Total\\\\Nitrogen}", "Nitrates")

# function to filter and impute missing values for given variable
filter_impute <- function(x, variable) {
    variable_sym <- sym(variable)

    x %>%
        group_by(station) %>%
        filter(sum(is.na(!!variable_sym)) / n() <= .05) %>%
        mutate(across(all_of(variable), ~ zoo::na.approx(., na.rm = FALSE))) %>%
        ungroup()
}

###
# main result
###

# prepare analysis grid
analysis_results <- tibble(
    dep_var = dep_vars
)

analysis_results <- analysis_results %>%
    mutate(
        formula = map(dep_var, ~ as.formula(paste0(.x, " ~ I(1 - d_forest_share) | station + year | 0 | estuary + estuary_year"))),
        model = map2(formula, dep_var, ~ felm(.x, data = analysis_panel %>% filter(year >= 2000) %>% filter_impute(.y)))
    )
texreg(analysis_results$model,
    file = "output/tables/reg_stations_deforestation.tex",
    custom.model.names = dep_vars_table,
    custom.coef.names = c("Deforestation"),
    stars = c(0.01, 0.05, 0.1),
    caption = "Regression Results: Aggregate Deforestation and Pollution",
    label = "tbl-reg-deforestation-sensors",
    include.rsquared = FALSE, include.adjrs = FALSE,
    custom.note = "\\item \\textit{Notes:} Unbalanced panels of all available data post 2000. Models include station and year fixed effects. Standard errors are clustered at the watershed and watershed-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = 0.8,
    dcolumn = TRUE
)

htmlreg(
    file = "output/presentations/final_seminar/reg_stations_deforestation.html",
    analysis_results$model[c(1, 3, 5, 7)],
    custom.model.names = c("pH", "BOD5.20", "Total Residue", "Nitrates"),
    custom.coef.names = c("Deforestation"),
    caption = "",
    stars = c(0.01, 0.05, 0.1),
    padding = 12,    
    include.rsquared = FALSE, include.adjrs = FALSE,
)

analysis_results <- analysis_results %>%
    mutate(
        formula = map(dep_var, ~ as.formula(paste0(.x, "_sd ~ I(1 - d_forest_share) | station + year | 0 | estuary + estuary_year"))),
        model = map2(formula, dep_var, ~ felm(.x, data = analysis_panel %>% filter(year >= 2000) %>% filter_impute(.y)))
    )

texreg(analysis_results$model,
    file = "output/tables/reg_stations_deforestation_sd.tex",
    custom.model.names = dep_vars_table,
    custom.coef.names = c("Deforestation"),
    stars = c(0.01, 0.05, 0.1),
    caption = "Regression Results: Deforestation and Pollution, Standardized Dependent Variables",
    label = "tbl-reg-deforestation-sensors-sd",
    include.rsquared = FALSE, include.adjrs = FALSE,
    custom.note = "\\item \\textit{Notes:} Unbalanced panels of all available data post 2000. Models include station and year fixed effects. Standard errors are clustered at the watershed and watershed-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = 0.8,
    dcolumn = TRUE
)

###
# Effect of replacement
###

analysis_results <- analysis_results %>%
    mutate(
        formula = map(dep_var, ~ as.formula(paste0(.x, " ~ d_pasture_share + d_agriculture_share + d_urban_share + d_mining_share | station + year | 0 | estuary + estuary_year"))),
        model = map2(formula, dep_var, ~ felm(.x, data = analysis_panel %>% filter(year >= 2000) %>% filter_impute(.y)))
    )

texreg(analysis_results$model,
    file = "output/tables/reg_stations_deforestation_replacement.tex",
    custom.model.names = dep_vars_table,
    custom.coef.names = c("$\\Delta$Pasture", "$\\Delta$Agriculture", "$\\Delta$Urban", "$\\Delta$Mining"),
    stars = c(0.01, 0.05, 0.1),
    caption = "Regression Results: Deforestation and Pollution, Replacement Effects",
    label = "tbl-reg-deforestation-sensors-replacement",
    include.rsquared = FALSE, include.adjrs = FALSE,
    custom.note = "\\item \\textit{Notes:} Unbalanced panels of all available data post 2000. Models include station and year fixed effects. Standard errors are clustered at the watershed and watershed-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = 0.7,
    dcolumn = TRUE
)

###
# Levels
###

analysis_results <- analysis_results %>%
    mutate(
        formula = map(dep_var, ~ as.formula(paste0(.x, " ~ forest_share + pasture_share + agriculture_share + urban_share + mining_share | station + year | 0 | estuary + estuary_year"))),
        model = map2(formula, dep_var, ~ felm(.x, data = analysis_panel %>% filter(year >= 2000) %>% filter_impute(.y)))
    )

texreg(analysis_results$model,
    file = "output/tables/reg_stations_deforestation_replacement_levels.tex",
    custom.model.names = dep_vars_table,
    custom.coef.names = c("$\\rho$Forest", "$\\rho$Pasture", "$\\rho$Agriculture", "$\\rho$Urban", "$\\rho$Mining"),
    stars = c(0.01, 0.05, 0.1),
    caption = "Regression Results: Deforestation and Pollution, Replacement Effects, Levels",
    label = "tbl-reg-deforestation-sensors-replacement-levels",
    include.rsquared = FALSE, include.adjrs = FALSE,
    custom.note = "\\item \\textit{Notes:} Unbalanced panels of all available data post 2000. Models include station and year fixed effects. Standard errors are clustered at the watershed and watershed-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = 0.7,
    dcolumn = TRUE
)

###
# Distance
###

# prepare analysis grid
analysis_results <- tibble(
    dep_var = dep_vars
)

analysis_results <- analysis_results %>%
    mutate(
        formula = map(dep_var, ~ as.formula(paste0(.x, " ~ d_forest_share + d_forest_share : I(distance / 100) | station + year | 0 | estuary + estuary_year"))),
        model = map2(formula, dep_var, ~ felm(.x, data = analysis_bins %>% filter(year > 2000) %>% filter_impute(.y)))
    )

texreg(analysis_results$model,
    file = "output/tables/reg_stations_deforestation_distance.tex",
    custom.model.names = dep_vars_table,
    custom.coef.names = c("Deforestation", "Deforestation x Distance"),
    stars = c(0.01, 0.05, 0.1),
    caption = "Regression Results: Deforestation and Pollution, Effect in Distance",
    label = "tbl-reg-deforestation-sensors-distance",
    include.rsquared = FALSE, include.adjrs = FALSE,
    custom.note = "\\item \\textit{Notes:} Unbalanced panels of all available data post 2000. Models include station fixed effects. Standard errors are clustered at the watershed and watershed-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = 0.8,
    dcolumn = TRUE
)

###########################################
#                                         #
#             HEALTH OUTCOMES             #
#                                         #
###########################################


###
# Load data
###

large_panel <- read_csv("data/analysis/large_panel.csv")
small_panel <- read_csv("data/analysis/small_panel.csv")
large_panel_l1 <- read_csv("data/analysis/large_panel_l1.csv")
small_panel_l1 <- read_csv("data/analysis/small_panel_l1.csv")
census_panel <- read_csv("data/analysis/census_panel.csv")
placebo_panel <- read_csv("data/analysis/placebo_panel.csv")

###
# First stage, Deter
###

first_stage_deter <- list()
first_stage_deter[["A"]] <- felm(forest_upstream_share ~ cloud_cover_DETER_cum_upstream | municipality + year | 0 | municipality + region_year, data = large_panel)
first_stage_deter[["B"]] <- felm(forest_upstream_share ~ cloud_cover_DETER_cum_upstream + temperature + precipitation | municipality + year | 0 | municipality + region_year, data = large_panel)
first_stage_deter[["C"]] <- felm(forest_upstream_share ~ cloud_cover_DETER_cum_upstream + temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | 0 | municipality + region_year, data = large_panel)
first_stage_deter[["D"]] <- felm(forest_upstream_share ~ cloud_cover_DETER_cum_upstream + temperature + precipitation + gdp_pc + educ_ideb + vaccination_index_5y + health_primary_care_coverage + health_doctors_1000 | municipality + year | 0 | municipality + region_year, data = small_panel)

texreg(first_stage_deter,
    file = "output/tables/reg_forest_DETER_first_stage.tex",
    custom.header = list("Forest Cover" = 1:4),
    custom.coef.names = c("Cumulative Cloud Cover"),
    omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 4,
    caption = "OLS Regression Results: First stage results",
    label = "tbl-reg-deter-first-stage",
    include.rsquared = FALSE, include.adjrs = FALSE,
    custom.gof.rows = list(
        "F-statistic" = map_dbl(first_stage_deter, ~ .x %>%
            summary() %>%
            pluck("P.fstat") %>%
            pluck("F")),
        "Climate Controls" = list("", "\\ding{51}", "\\ding{51}", "\\ding{51}"),
        "Extended Controls" = list("", "", "\\ding{51}", "\\ding{51}"),
        "Full Controls" = list("", "", "", "\\ding{51}")
    ),
    custom.note = "\\item \\textit{Notes:} I regress the weighted share of area covered with forests on the weighted upstream DETER cloud cover. Controls are added consecutively in groups. Climate Controls include precipitation and temperature. Extended Controls add GDP per capita, educational scores, and a vaccination index for waterborne diseases. Full controls are the number of doctors per 1000 inhabitants and primary health care coverage (with a reduced panel starting in 2010). Models include municipality and year fixed effects. Standard errors are clustered at the municipalitiy and region-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = .9,
    dcolumn = TRUE
)

# ## Additional diagnostics

# ## Calculate standard deviation of the dependent variable
# large_panel %>% summarise(sd(1 - forest_upstream_share_d)) %>% pull()
# small_panel %>% summarise(sd(1 - forest_upstream_share_d)) %>% pull()

# ## Calculate VIF for model with controls
# lm_model_with_controls <- lm(I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_upstream + temperature + precipitation, data = large_panel)
# car::vif(lm_model_with_controls)

# ## Calculate partial R-squared for model with controls
# # Regress the endogenous variable on controls only to get residuals
# controls_model <- felm(I(1 - forest_upstream_share_d) ~ temperature + precipitation | municipality + year | 0 | municipality + region_year, data = large_panel)
# residuals_controls <- controls_model$residuals
# # Regress the residuals on the instrument
# instrument_model <- felm(residuals_controls ~ cloud_cover_DETER_upstream | 0 | 0 | municipality + region_year, data = large_panel)
# partial_r2_with_controls <- summary(instrument_model)$r.squared
# partial_r2_without_controls <- summary(first_stage_deter[["A"]])$P.r.squared

# # Calculate the partial R-squared
# partial_r2_with_controls / partial_r2_without_controls

###
# Second stage, Deter, Total Mortality
###

second_stage_deter_mortality_tot <- list()
second_stage_deter_mortality_tot[["OLS-B"]] <- felm(I(mortality_rate_tot * 1000) ~ forest_upstream_share + temperature + precipitation | municipality + year | 0 | municipality + region_year, data = large_panel)
second_stage_deter_mortality_tot[["A"]] <- felm(I(mortality_rate_tot * 1000) ~ 1 | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel)
second_stage_deter_mortality_tot[["B"]] <- felm(I(mortality_rate_tot * 1000) ~ temperature + precipitation | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel)
second_stage_deter_mortality_tot[["C"]] <- felm(I(mortality_rate_tot * 1000) ~ temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel)
second_stage_deter_mortality_tot[["D"]] <- felm(I(mortality_rate_tot * 1000) ~ temperature + precipitation + gdp_pc + educ_ideb + vaccination_index_5y + health_primary_care_coverage + health_doctors_1000 | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = small_panel)

texreg(
    second_stage_deter_mortality_tot,
    file = "output/tables/reg_forest_DETER_mortality_tot.tex",
    custom.header = list("OLS" = 1, "IV" = 2:5),
    custom.model.names = c("B", "A", "B", "C", "D"),
    custom.coef.names = c("Forest Cover", "$\\widehat{\\text{Forest Cover}}$"),
    omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 3,
    caption = "IV Regression Results: Upstream Forest Cover and Mortality (per 1,000 inhabitants)",
    label = "tbl-reg-deter-mortality-tot",
    include.rsquared = FALSE, include.adjrs = FALSE,
    custom.gof.rows = list(
        "Climate Controls" = list("\\ding{51}", "", "\\ding{51}", "\\ding{51}", "\\ding{51}"),
        "Extended Controls" = list("", "", "", "\\ding{51}", "\\ding{51}"),
        "Full Controls" = list("", "", "", "", "\\ding{51}")
    ),
    custom.note = "\\item \\textit{Notes:} I regress the overall mortality rate on the instrumented weighted proportion of upstream area covered with forests. Controls are added consecutively in groups. Climate Controls include precipitation and temperature. Extended Controls add GDP per capita, educational scores, and a vaccination index for waterborne diseases. Full controls are the number of doctors per 1000 inhabitants and primary health care coverage (with a reduced panel starting in 2010). Models include municipality and year fixed effects. Standard errors are clustered at the municipalitiy and region-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = .9,
    dcolumn = TRUE
)

###
# Second stage, Deter, Infant Mortality
###

second_stage_deter_mortality_l1 <- list()
second_stage_deter_mortality_l1[["OLS-B"]] <- felm(mortality_rate_l1 ~ forest_upstream_share + temperature + precipitation | municipality + year | 0 | municipality + region_year, data = large_panel_l1)
second_stage_deter_mortality_l1[["A"]] <- felm(mortality_rate_l1 ~ 1 | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel_l1)
second_stage_deter_mortality_l1[["B"]] <- felm(mortality_rate_l1 ~ temperature + precipitation | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel_l1)
second_stage_deter_mortality_l1[["C"]] <- felm(mortality_rate_l1 ~ temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel_l1)
second_stage_deter_mortality_l1[["D"]] <- felm(mortality_rate_l1 ~ temperature + precipitation + gdp_pc + educ_ideb + vaccination_index_5y + health_primary_care_coverage + health_doctors_1000 | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = small_panel_l1)

texreg(
    second_stage_deter_mortality_l1,
    file = "output/tables/reg_forest_DETER_mortality_l1.tex",
    custom.header = list("OLS" = 1, "IV" = 2:5),
    custom.model.names = c("B", "A", "B", "C", "D"),
    custom.coef.names = c("Forest Cover", "$\\widehat{\\text{Forest Cover}}$"),
    omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 3,
    caption = "IV Regression Results: Upstream Forest Cover and Infant Mortality (per 1,000 births)",
    label = "tbl-reg-deter-mortality-l1",
    include.rsquared = FALSE, include.adjrs = FALSE,
    custom.gof.rows = list(
        "Climate Controls" = list("\\ding{51}", "", "\\ding{51}", "\\ding{51}", "\\ding{51}"),
        "Extended Controls" = list("", "", "", "\\ding{51}", "\\ding{51}"),
        "Full Controls" = list("", "", "", "", "\\ding{51}")
    ),
    custom.note = "\\item \\textit{Notes:} I regress the infant mortality rate on the instrumented weighted proportion of upstream area covered with forests. Controls are added consecutively in groups. Climate Controls include precipitation and temperature. Extended Controls add GDP per capita, educational scores, and a vaccination index for waterborne diseases. Full controls are the number of doctors per 1000 inhabitants and primary health care coverage (with a reduced panel starting in 2010). Models include municipality and year fixed effects. Standard errors are clustered at the municipalitiy and region-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = .9,
)



###
# Second stage, Deter, Hospitalizations
###

second_stage_deter_hosp_rate <- list()
second_stage_deter_hosp_rate[["OLS-B"]] <- felm(hosp_rate ~ forest_upstream_share + temperature + precipitation | municipality + year | 0 | municipality + region_year, data = large_panel)
second_stage_deter_hosp_rate[["A"]] <- felm(hosp_rate ~ 1 | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel)
second_stage_deter_hosp_rate[["B"]] <- felm(hosp_rate ~ temperature + precipitation | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel)
second_stage_deter_hosp_rate[["C"]] <- felm(hosp_rate ~ temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel)
second_stage_deter_hosp_rate[["D"]] <- felm(hosp_rate ~ temperature + precipitation + gdp_pc + educ_ideb + vaccination_index_5y + health_primary_care_coverage + health_doctors_1000 | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = small_panel)

texreg(
    second_stage_deter_hosp_rate,
    file = "output/tables/reg_forest_DETER_hosp_rate.tex",
    custom.header = list("OLS" = 1, "IV" = 2:5),
    custom.model.names = c("B", "A", "B", "C", "D"),
    custom.coef.names = c("Forest Cover", "$\\widehat{\\text{Forest Cover}}$"),
    omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 3,
    caption = "IV Regression Results: Upstream Forest Cover and Hospitalizations per Inhabitant",
    label = "tbl-reg-deter-hosp-rate",
    include.rsquared = FALSE, include.adjrs = FALSE,
    custom.gof.rows = list(
        "Climate Controls" = list("\\ding{51}", "", "\\ding{51}", "\\ding{51}", "\\ding{51}"),
        "Extended Controls" = list("", "", "", "\\ding{51}", "\\ding{51}"),
        "Full Controls" = list("", "", "", "", "\\ding{51}")
    ),
    custom.note = "\\item \\textit{Notes:} I regress the hospitalization rate on the instrumented weighted proportion of upstream area covered with forests. Controls are added consecutively in groups. Climate Controls include precipitation and temperature. Extended Controls add GDP per capita, educational scores, and a vaccination index for waterborne diseases. Full controls are the number of doctors per 1000 inhabitants and primary health care coverage (with a reduced panel starting in 2010). Models include municipality and year fixed effects. Standard errors are clustered at the municipalitiy and region-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = .9,
)

# standardized

second_stage_deter_hosp_rate_sd <- list()
second_stage_deter_hosp_rate_sd[["OLS-B"]] <- felm(hosp_rate_sd ~ forest_upstream_share + temperature + precipitation | municipality + year | 0 | municipality + region_year, data = large_panel %>% mutate(hosp_rate_sd = scale(hosp_rate)))
second_stage_deter_hosp_rate_sd[["A"]] <- felm(hosp_rate_sd ~ 1 | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel %>% mutate(hosp_rate_sd = scale(hosp_rate)))
second_stage_deter_hosp_rate_sd[["B"]] <- felm(hosp_rate_sd ~ temperature + precipitation | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel %>% mutate(hosp_rate_sd = scale(hosp_rate)))
second_stage_deter_hosp_rate_sd[["C"]] <- felm(hosp_rate_sd ~ temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel %>% mutate(hosp_rate_sd = scale(hosp_rate)))
second_stage_deter_hosp_rate_sd[["D"]] <- felm(hosp_rate_sd ~ temperature + precipitation + gdp_pc + educ_ideb + vaccination_index_5y + health_primary_care_coverage + health_doctors_1000 | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = small_panel %>% mutate(hosp_rate_sd = scale(hosp_rate)))

texreg(
    second_stage_deter_hosp_rate_sd,
    file = "output/tables/reg_forest_DETER_hosp_rate_sd.tex",
    custom.header = list("OLS" = 1, "IV" = 2:5),
    custom.model.names = c("B", "A", "B", "C", "D"),
    custom.coef.names = c("Forest Cover", "$\\widehat{\\text{Forest Cover}}$"),
    omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 3,
    caption = "IV Regression Results: Upstream Forest Cover and Hospitalizations per Inhabitant (Standardized)",
    label = "tbl-reg-deter-hosp-rate-sd",
    include.rsquared = FALSE, include.adjrs = FALSE,
    custom.gof.rows = list(
        "Climate Controls" = list("\\ding{51}", "", "\\ding{51}", "\\ding{51}", "\\ding{51}"),
        "Extended Controls" = list("", "", "", "\\ding{51}", "\\ding{51}"),
        "Full Controls" = list("", "", "", "", "\\ding{51}")
    ),
    custom.note = "\\item \\textit{Notes:} I regress the standarized hospitalization rate on the instrumented weighted proportion of upstream area covered with forests. Controls are added consecutively in groups. Climate Controls include precipitation and temperature. Extended Controls add GDP per capita, educational scores, and a vaccination index for waterborne diseases. Full controls are the number of doctors per 1000 inhabitants and primary health care coverage (with a reduced panel starting in 2010). Models include municipality and year fixed effects. Standard errors are clustered at the municipalitiy and region-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = .9,
)

###
# Second stage, Deter, Expenditure per Inhabitant
###

second_stage_deter_ex_pop <- list()
second_stage_deter_ex_pop[["OLS-B"]] <- felm(ex_pop ~ forest_upstream_share + temperature + precipitation | municipality + year | 0 | municipality + region_year, data = large_panel)
second_stage_deter_ex_pop[["A"]] <- felm(ex_pop ~ 1 | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel)
second_stage_deter_ex_pop[["B"]] <- felm(ex_pop ~ temperature + precipitation | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel)
second_stage_deter_ex_pop[["C"]] <- felm(ex_pop ~ temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel)
second_stage_deter_ex_pop[["D"]] <- felm(ex_pop ~ temperature + precipitation + gdp_pc + educ_ideb + vaccination_index_5y + health_primary_care_coverage + health_doctors_1000 | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = small_panel)

texreg(
    second_stage_deter_ex_pop,
    file = "output/tables/reg_forest_DETER_ex_pop.tex",
    custom.header = list("OLS" = 1, "IV" = 2:5),
    custom.model.names = c("B", "A", "B", "C", "D"),
    custom.coef.names = c("Forest Cover", "$\\widehat{\\text{Forest Cover}}$"),
    omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 3,
    caption = "IV Regression Results: Upstream Forest Cover and Medical Expenditure per Inhabitant",
    label = "tbl-reg-deter-ex-pop",
    include.rsquared = FALSE, include.adjrs = FALSE,
    custom.gof.rows = list(
        "Climate Controls" = list("\\ding{51}", "", "\\ding{51}", "\\ding{51}", "\\ding{51}"),
        "Extended Controls" = list("", "", "", "\\ding{51}", "\\ding{51}"),
        "Full Controls" = list("", "", "", "", "\\ding{51}")
    ),
    custom.note = "\\item \\textit{Notes:} I regress the medical expenditure per inhabitant on the instrumented weighted proportion of upstream area covered with forests. Controls are added consecutively in groups. Climate Controls include precipitation and temperature. Extended Controls add GDP per capita, educational scores, and a vaccination index for waterborne diseases. Full controls are the number of doctors per 1000 inhabitants and primary health care coverage (with a reduced panel starting in 2010). Models include municipality and year fixed effects. Standard errors are clustered at the municipalitiy and region-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = .9,
)

htmlreg(
    list(
        "Total Mortality" = second_stage_deter_mortality_tot[["B"]],
        "Infant Mortality" = second_stage_deter_mortality_l1[["B"]],
        "Hospitalization Rate" = second_stage_deter_hosp_rate[["B"]],
        "Expenditure per Inhabitant" = second_stage_deter_ex_pop[["B"]]
    ),
    file = "output/presentations/final_seminar/reg_forest_DETER_all.html",
    custom.coef.names = c("$\\widehat{\\text{Forest Cover}}$"),
    caption = "",
    omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 3,
    padding = 12,
    include.rsquared = FALSE, include.adjrs = FALSE,
)



###
# Regional and temporal subsets
###

second_stage_deter_hosp_rate_subset <- list()
second_stage_deter_hosp_rate_subset[["B-N"]] <- large_panel %>%
    filter(region == "North") %>%
    felm(hosp_rate ~ temperature + precipitation | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = .)
second_stage_deter_hosp_rate_subset[["B-NE"]] <- large_panel %>%
    filter(region == "Northeast") %>%
    felm(hosp_rate ~ temperature + precipitation | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = .)
second_stage_deter_hosp_rate_subset[["B-CW"]] <- large_panel %>%
    filter(region == "Central-West") %>%
    felm(hosp_rate ~ temperature + precipitation | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = .)
second_stage_deter_hosp_rate_subset[["B-08"]] <- large_panel %>%
    filter(year %in% 2005:2008) %>%
    felm(hosp_rate ~ temperature + precipitation | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = .)
second_stage_deter_hosp_rate_subset[["B-12"]] <- large_panel %>%
    filter(year %in% 2009:2012) %>%
    felm(hosp_rate ~ temperature + precipitation | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = .)
second_stage_deter_hosp_rate_subset[["B-17"]] <- large_panel %>%
    filter(year %in% 2013:2017) %>%
    felm(hosp_rate ~ temperature + precipitation | municipality + year | (forest_upstream_share ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = .)

texreg(
    second_stage_deter_hosp_rate_subset,
    file = "output/tables/reg_forest_DETER_hosp_rate_subsets.tex",
    custom.header = list("OLS, C" = 1:6), # custom.header = list("Expenditure per Inhabitant" = 1:5),
    custom.model.names = c("North", "Northeast", "Central-West", "2005-2008", "2009-2012", "2013-2017"),
    custom.coef.names = c("$\\widehat{\\text{Forest Cover}}$"),
    omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 1,
    caption = "IV Regression Results: Upstream Forest Cover and Hospitalizations per Inhabitant, Subsets",
    label = "tbl-reg-deter-hosp-rate-subsets",
    include.rsquared = FALSE, include.adjrs = FALSE,
    custom.gof.rows = list(
        "First stage F-statistic" = map_dbl(second_stage_deter_hosp_rate_subset, ~ .x$stage1$iv1fstat$forest_upstream_share[["F"]]),
        "Climate Controls" = list("\\ding{51}", "\\ding{51}", "\\ding{51}", "\\ding{51}", "\\ding{51}", "\\ding{51}")
    ),
    custom.note = "\\item \\textit{Notes:} I regress the medical expenditure rate on the instrumented weighted upstream deforestation rate. Controls are added consecutively in groups. Climate Controls include precipitation and temperature. Extended Controls add GDP per capita, educational scores, and a vaccination index for waterborne diseases. Full controls are the number of doctors per 1000 inhabitants and primary health care coverage (with a reduced panel starting in 2010). Models include municipality and year fixed effects. Standard errors are clustered at the municipalitiy and region-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = .8,
    dcolumn = TRUE
)

###
# OLS, Migration from Census
###

ols_migration <- list()
ols_migration[["\\makecell{Total\\\\Mortality}"]] <- felm(I(mortality_rate_tot * 1000) ~ forest_upstream_share + temperature + precipitation + pop_from_municipality + pop_white | municipality + year | 0 | municipality + region_year, data = census_panel)
ols_migration[["\\makecell{Infant\\\\Mortality}"]] <- felm(mortality_rate_l1 ~ forest_upstream_share + temperature + precipitation + pop_from_municipality + pop_white | municipality + year | 0 | municipality + region_year, data = census_panel)
ols_migration[["\\makecell{Hospitalization\\\\Rate}"]] <- felm(hosp_rate ~ forest_upstream_share + temperature + precipitation + pop_from_municipality + pop_white | municipality + year | 0 | municipality + region_year, data = census_panel)
ols_migration[["\\makecell{Expenditure\\\\per Inhabitant}"]] <- felm(ex_pop ~ forest_upstream_share + temperature + precipitation + pop_from_municipality + pop_white | municipality + year | 0 | municipality + region_year, data = census_panel)

texreg(
    ols_migration,
    file = "output/tables/reg_migration.tex",
    # custom.header = list("Total Mortality" = 1:3),
    custom.coef.names = c("Forest Cover", "Born Local", "White"),
    omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 3,
    caption = "OLS Regression Results: Migration and Health Outcomes",
    label = "tbl-reg-migration",
    include.rsquared = FALSE, include.adjrs = FALSE,
    custom.gof.rows = list(
        "Climate Controls" = list("\\ding{51}", "\\ding{51}", "\\ding{51}", "\\ding{51}")
    ),
    custom.note = "\\item \\textit{Notes:} I regress health outcomes on the upstream deforestation rate, the number of migrants from the municipality, and the number of white inhabitants. Controls are precipitation and temperature. Models include municipality and year fixed effects. Standard errors are clustered at the municipalitiy and region-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = .9,
    dcolumn = TRUE
)

###
# OLS, Placebo
###

ols_placebo <- list()
ols_placebo[["DETER"]] <- felm(cloud_cover_DETER ~ cloud_cover | municipality + year | 0 | municipality + year, data = placebo_panel)
ols_placebo[["pre_DETER"]] <- felm(I((1 - forest_d_rate) * 100) ~ cloud_cover + cloud_cover:legal_amazon | municipality + year | 0 | municipality + year, data = placebo_panel %>% filter(!DETER_active))
ols_placebo[["outside_legal_amazon"]] <- felm(I((1 - forest_d_rate) * 100) ~ cloud_cover + cloud_cover:legal_amazon | municipality + year | 0 | municipality + year, data = placebo_panel %>% filter(DETER_active))

texreg(
    ols_placebo,
    file = "output/tables/reg_placebo.tex",
    custom.header = list("\\makecell{DETER\\\\Cloud Cover}" = 1, "Deforestation Rate" = 2:3),
    custom.model.names = c("", "pre-2005", "post-2005"),
    custom.coef.names = c("ESA Cloud Cover", "ESA Cloud Cover $x$ DETER area"),
    # omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 3,
    caption = "OLS Regression Results: Placebo Test with ESA Cloud Cover data",
    label = "tbl-reg-placebo",
    include.rsquared = FALSE, include.adjrs = FALSE,
    # custom.gof.rows = list(
    #     "Climate Controls" = list("\\ding{51}", "\\ding{51}", "\\ding{51}", "\\ding{51}")
    # ),
    custom.note = "\\item \\textit{Notes:} I regress DETER cloud cover and Deforestation on ESA cloud cover. The latter two columns restrict the sample to the years up to and after 2004, respectively. Models include municipality and year fixed effects. Standard errors are clustered at the municipalitiy and year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = .9,
    dcolumn = TRUE
)

###
# Spatial Regression
###

spatial_weights <- readMM("data/misc/municipality_weights.mtx")
colnames(spatial_weights) <- read_csv("data/misc/municipality_weights.csv") %>% pull()
rownames(spatial_weights) <- read_csv("data/misc/municipality_weights.csv") %>% pull()
spatial_weights_subset <- spatial_weights[large_panel$municipality %>%
    unique() %>%
    as.character(), large_panel$municipality %>% unique() %>% as.character()]

spatial_weights_subset <- mat2listw(spatial_weights_subset, style = "W")

## Test for spatial autocorrelation

# Define the tests
spatial_tests <- list(
    "Total Mortality" = mortality_rate_tot ~ forest_upstream_share + temperature + precipitation,
    "Hospitalization Rate" = hosp_rate ~ forest_upstream_share + temperature + precipitation,
    "Expenditure per Inhabitant" = ex_pop ~ forest_upstream_share + temperature + precipitation
)

# Run the tests
spatial_tests <- spatial_tests %>% map(
    ~ list(
        "RLML" = slmtest(
            .x,
            data = large_panel, listw = spatial_weights_subset, test = "rlml", model = "within"
        ),
        "RLME" = slmtest(
            .x,
            data = large_panel, listw = spatial_weights_subset, test = "rlme", model = "within"
        )
    )
)

# Extract the test statistics
spatial_tests_table <- spatial_tests %>%
    map(
        ~ map(
            .x,
            ~ .x %>%
                pluck("statistic") %>%
                as.double()
        ) %>%
            bind_rows()
    ) %>%
    bind_rows(.id = "Outcome")

# Generate the LaTeX table string
latex_table <- spatial_tests_table %>%
    mutate(across(where(is.numeric), ~ sprintf("%.2f", .))) %>%
    kableExtra::kbl(
        format = "latex", booktabs = TRUE,
        caption = "Tests for Spatial Autocorrelation", linesep = ""
    ) %>%
    as.character()

# Insert the \label command after the third line
lines <- strsplit(latex_table, "\n")[[1]]
lines <- append(lines, "\\label{tbl-spatial-tests}", after = 3)
latex_table_with_label <- paste(lines, collapse = "\n")

# Write the modified LaTeX table string to a file
output_file <- "output/tables/spatial_tests.tex"
writeLines(latex_table_with_label, output_file)

## Spatial regression, no IV

spatial_regression <- list()
spatial_regression[["Total Mortality"]] <- spml(
    mortality_rate_tot ~ forest_upstream_share + temperature + precipitation,
    listw = spatial_weights_subset, spatial.error = "b", effect = "twoways", data = large_panel
)
spatial_regression[["Hospitalization Rate"]] <- spml(
    hosp_rate ~ forest_upstream_share + temperature + precipitation,
    listw = spatial_weights_subset, spatial.error = "b", effect = "twoways", data = large_panel
)
spatial_regression[["Expenditure per Inhabitant"]] <- spml(
    ex_pop ~ forest_upstream_share + temperature + precipitation,
    listw = spatial_weights_subset, spatial.error = "b", effect = "twoways", data = large_panel
)

source("code/analysis/extract.splm.R")
spatial_regression <- spatial_regression %>% map(extract.splm)

## Spatial regression, IV

# First-stage regression
first_stage <- felm(
    forest_upstream_share ~ cloud_cover_DETER_cum_upstream + temperature + precipitation | municipality + year | 0 | municipality + region_year,
    data = large_panel
)

# Get the fitted values (instrumented variable)
large_panel$forest_upstream_share_iv <- fitted(first_stage) %>% as.vector()

spatial_regression_iv <- list()
spatial_regression_iv[["Total Mortality"]] <- spml(
    mortality_rate_tot ~ forest_upstream_share_iv + temperature + precipitation,
    listw = spatial_weights_subset, spatial.error = "b", effect = "twoways", data = large_panel
)
spatial_regression_iv[["Hospitalization Rate"]] <- spml(
    hosp_rate ~ forest_upstream_share_iv + temperature + precipitation,
    listw = spatial_weights_subset, spatial.error = "b", effect = "twoways", data = large_panel
)
spatial_regression_iv[["Expenditure per Inhabitant"]] <- spml(
    ex_pop ~ forest_upstream_share_iv + temperature + precipitation,
    listw = spatial_weights_subset, spatial.error = "b", effect = "twoways", data = large_panel
)

spatial_regression_iv <- spatial_regression_iv %>% map(extract.splm)

texreg(
    c(spatial_regression, spatial_regression_iv),
    file = "output/tables/reg_spatial.tex",
    custom.header = list(" " = 1:3, "IV" = 4:6),
    custom.model.names = c("\\makecell{Total\\\\Mortality}", "\\makecell{Hospitalization\\\\Rate}", "\\makecell{Expenditure\\\\per Inhabitant}", "\\makecell{Total\\\\Mortality}", "\\makecell{Hospitalization\\\\Rate}", "\\makecell{Expenditure\\\\per Inhabitant}"),
    custom.coef.names = c("Spatial $\\rho$", "Forest Cover", "$\\widehat{\\text{Forest Cover}}$"),
    omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 3,
    caption = "Spatial Panel Model: Health Outcomes and Deforestation with Spatial Error",
    label = "tbl-reg-spatial",
    include.loglik = FALSE,
    custom.gof.rows = list(
        "Climate Controls" = rep_len("\\ding{51}", 6)
    ),
    custom.note = "\\item \\textit{Notes:} I estimate Maximum Likelihood Spatial Error Data Models with a Within Transformation. Health Outcomes are regressed on (instrumented) Forest Cover. Models include climate controls, municipality and region-year fixed effects. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = .7,
    dcolumn = TRUE
)