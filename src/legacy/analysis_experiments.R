first_stage_deter_d <- list()
first_stage_deter_d[["A"]] <- felm(I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_upstream | municipality + year | 0 | municipality + region_year, data = large_panel)
first_stage_deter_d[["B"]] <- felm(I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_upstream + temperature + precipitation | municipality + year | 0 | municipality + region_year, data = large_panel)
first_stage_deter_d[["C"]] <- felm(I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_upstream + temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | 0 | municipality + region_year, data = large_panel)
first_stage_deter_d[["D"]] <- felm(I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_upstream + temperature + precipitation + gdp_pc + educ_ideb + vaccination_index_5y + health_primary_care_coverage + health_doctors_1000 | municipality + year | 0 | municipality + region_year, data = small_panel)

texreg(first_stage_deter_d,
    file = "output/tables/reg_deforestation_DETER_first_stage.tex",
    custom.header = list("Deforestation" = 1:4),
    custom.coef.names = c("Cloud Cover", "", "", "", "", "", "", ""),
    omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 4,
    caption = "OLS Regression Results: First stage results",
    label = "tbl-reg-deter-first-stage-d",
    include.rsquared = FALSE, include.adjrs = FALSE,
    custom.gof.rows = list(
        "F-statistic" = map_dbl(first_stage_deter_d, ~ .x %>%
            summary() %>%
            pluck("P.fstat") %>%
            pluck("F")),
        "Climate Controls" = list("", "\\ding{51}", "\\ding{51}", "\\ding{51}"),
        "Extended Controls" = list("", "", "\\ding{51}", "\\ding{51}"),
        "Full Controls" = list("", "", "", "\\ding{51}")
    ),
    custom.note = "\\item \\textit{Notes:} I regress the weighted upstream deforestation rate on the weighted upstream DETER cloud cover. Controls are added consecutively in groups. Climate Controls include precipitation and temperature. Extended Controls add GDP per capita, educational scores, a vaccination index for waterborne diseases. Full controls are the number of doctors per 1000 inhabitants and primary health care coverage (with a reduced panel starting in 2010). Models include municipality and year fixed effects. Standard errors are clustered at the municipalitiy and region-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = .9,
    dcolumn = TRUE
)

second_stage_deter_mortality_tot_d <- list()
second_stage_deter_mortality_tot_d[["OLS-B"]] <- felm(I(mortality_rate_tot * 1000) ~ I(1 - forest_upstream_share_d) + temperature + precipitation | municipality + year | 0 | municipality + region_year, data = large_panel)
second_stage_deter_mortality_tot_d[["OLS-C"]] <- felm(I(mortality_rate_tot * 1000) ~ I(1 - forest_upstream_share_d) + temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | 0 | municipality + region_year, data = large_panel)
second_stage_deter_mortality_tot_d[["OLS-D"]] <- felm(I(mortality_rate_tot * 1000) ~ I(1 - forest_upstream_share_d) + temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb + health_primary_care_coverage + health_doctors_1000 | municipality + year | 0 | municipality + region_year, data = small_panel)
second_stage_deter_mortality_tot_d[["A"]] <- felm(I(mortality_rate_tot * 1000) ~ 1 | municipality + year | (I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_upstream) | municipality + region_year, data = large_panel)
second_stage_deter_mortality_tot_d[["B"]] <- felm(I(mortality_rate_tot * 1000) ~ temperature + precipitation | municipality + year | (I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_upstream) | municipality + region_year, data = large_panel)
second_stage_deter_mortality_tot_d[["C"]] <- felm(I(mortality_rate_tot * 1000) ~ temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | (I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_upstream) | municipality + region_year, data = large_panel)
second_stage_deter_mortality_tot_d[["D"]] <- felm(I(mortality_rate_tot * 1000) ~ temperature + precipitation + gdp_pc + educ_ideb + vaccination_index_5y + health_primary_care_coverage + health_doctors_1000 | municipality + year | (I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_upstream) | municipality + region_year, data = small_panel)

texreg(
    second_stage_deter_mortality_tot_d,
    file = "output/tables/reg_deforestation_DETER_mortality_tot.tex",
    custom.header = list("OLS" = 1:3, "IV" = 4:7), # custom.header = list("Total Mortality Rate" = 1:5),
    custom.model.names = c("B", "C", "D", "A", "B", "C", "D"),
    custom.coef.names = c("Deforestation", "$\\widehat{\\text{Deforestation}}$"),
    omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 3,
    caption = "IV Regression Results: Upstream Deforestation and Mortality (per 1,000 inhabitants)",
    label = "tbl-reg-deter-mortality-tot-d",
    include.rsquared = FALSE, include.adjrs = FALSE,
    custom.gof.rows = list(
        "Climate Controls" = list("\\ding{51}", "\\ding{51}", "\\ding{51}", "", "\\ding{51}", "\\ding{51}", "\\ding{51}"),
        "Extended Controls" = list("", "\\ding{51}", "\\ding{51}", "", "", "\\ding{51}", "\\ding{51}"),
        "Full Controls" = list("", "", "\\ding{51}", "", "", "", "\\ding{51}")
    ),
    custom.note = "\\item \\textit{Notes:} I regress the overall mortality rate on the instrumented weighted upstream deforestation rate. Controls are added consecutively in groups. Climate Controls include precipitation and temperature. Extended Controls add GDP per capita, educational scores, a vaccination index for waterborne diseases. Full controls are the number of doctors per 1000 inhabitants and primary health care coverage (with a reduced panel starting in 2010). Models include municipality and year fixed effects. Standard errors are clustered at the municipalitiy and region-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = .7,
    dcolumn = TRUE
)

second_stage_deter_mortality_l1_d <- list()
second_stage_deter_mortality_l1_d[["OLS-B"]] <- felm(mortality_rate_l1 ~ I(1 - forest_upstream_share_d) + temperature + precipitation | municipality + year | 0 | municipality + region_year, data = large_panel_l1)
second_stage_deter_mortality_l1_d[["OLS-C"]] <- felm(mortality_rate_l1 ~ I(1 - forest_upstream_share_d) + temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | 0 | municipality + region_year, data = large_panel_l1)
second_stage_deter_mortality_l1_d[["OLS-D"]] <- felm(mortality_rate_l1 ~ I(1 - forest_upstream_share_d) + temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb + health_primary_care_coverage + health_doctors_1000 | municipality + year | 0 | municipality + region_year, data = small_panel_l1)
second_stage_deter_mortality_l1_d[["A"]] <- felm(mortality_rate_l1 ~ 1 | municipality + year | (I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_upstream) | municipality + region_year, data = large_panel_l1)
second_stage_deter_mortality_l1_d[["B"]] <- felm(mortality_rate_l1 ~ temperature + precipitation | municipality + year | (I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_upstream) | municipality + region_year, data = large_panel_l1)
second_stage_deter_mortality_l1_d[["C"]] <- felm(mortality_rate_l1 ~ temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | (I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_upstream) | municipality + region_year, data = large_panel_l1)
second_stage_deter_mortality_l1_d[["D"]] <- felm(mortality_rate_l1 ~ temperature + precipitation + gdp_pc + educ_ideb + vaccination_index_5y + health_primary_care_coverage + health_doctors_1000 | municipality + year | (I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_upstream) | municipality + region_year, data = small_panel_l1)

texreg(
    second_stage_deter_mortality_l1_d,
    file = "output/tables/reg_deforestation_DETER_mortality_l1.tex",
    custom.header = list("OLS" = 1:3, "IV" = 4:7), # custom.header = list("Infant Mortality Rate" = 1:5),
    custom.model.names = c("B", "C", "D", "A", "B", "C", "D"),
    custom.coef.names = c("Deforestation", "$\\widehat{\\text{Deforestation}}$"),
    omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 1,
    caption = "IV Regression Results: Upstream Deforestation and Infant Mortality (per 1,000 births)",
    label = "tbl-reg-deter-mortality-l1-d",
    include.rsquared = FALSE, include.adjrs = FALSE,
    custom.gof.rows = list(
        "Climate Controls" = list("\\ding{51}", "\\ding{51}", "\\ding{51}", "", "\\ding{51}", "\\ding{51}", "\\ding{51}"),
        "Extended Controls" = list("", "\\ding{51}", "\\ding{51}", "", "", "\\ding{51}", "\\ding{51}"),
        "Full Controls" = list("", "", "\\ding{51}", "", "", "", "\\ding{51}")
    ),
    custom.note = "\\item \\textit{Notes:} I regress the infant mortality rate on the instrumented weighted upstream deforestation rate. Controls are added consecutively in groups. Climate Controls include precipitation and temperature. Extended Controls add GDP per capita, educational scores, a vaccination index for waterborne diseases. Full controls are the number of doctors per 1000 inhabitants and primary health care coverage (with a reduced panel starting in 2010). Models include municipality and year fixed effects. Standard errors are clustered at the municipalitiy and region-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = .7,
    dcolumn = TRUE
)

second_stage_deter_hosp_rate_d <- list()
second_stage_deter_hosp_rate_d[["OLS-B"]] <- felm(hosp_rate ~ I(1 - forest_upstream_share_d) + temperature + precipitation | municipality + year | 0 | municipality + region_year, data = large_panel)
second_stage_deter_hosp_rate_d[["OLS-C"]] <- felm(hosp_rate ~ I(1 - forest_upstream_share_d) + temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | 0 | municipality + region_year, data = large_panel)
second_stage_deter_hosp_rate_d[["OLS-D"]] <- felm(hosp_rate ~ I(1 - forest_upstream_share_d) + temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb + health_primary_care_coverage + health_doctors_1000 | municipality + year | 0 | municipality + region_year, data = small_panel)
second_stage_deter_hosp_rate_d[["A"]] <- felm(hosp_rate ~ 1 | municipality + year | (I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel)
second_stage_deter_hosp_rate_d[["B"]] <- felm(hosp_rate ~ temperature + precipitation | municipality + year | (I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel)
second_stage_deter_hosp_rate_d[["C"]] <- felm(hosp_rate ~ temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | (I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = large_panel)
second_stage_deter_hosp_rate_d[["D"]] <- felm(hosp_rate ~ temperature + precipitation + gdp_pc + educ_ideb + vaccination_index_5y + health_primary_care_coverage + health_doctors_1000 | municipality + year | (I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_cum_upstream) | municipality + region_year, data = small_panel)

texreg(
    second_stage_deter_hosp_rate_d,
    file = "output/tables/reg_deforestation_DETER_hosp_rate.tex",
    custom.header = list("OLS" = 1:3, "IV" = 4:7), # custom.header = list("Hospitalization Rate" = 1:5),
    custom.model.names = c("B", "C", "D", "A", "B", "C", "D"),
    custom.coef.names = c("Deforestation", "$\\widehat{\\text{Deforestation}}$"),
    omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 4,
    caption = "IV Regression Results: Upstream Deforestation and Hospitalizations per Inhabitant",
    label = "tbl-reg-deter-hosp-rate-d",
    include.rsquared = FALSE, include.adjrs = FALSE,
    custom.gof.rows = list(
        "Climate Controls" = list("\\ding{51}", "\\ding{51}", "\\ding{51}", "", "\\ding{51}", "\\ding{51}", "\\ding{51}"),
        "Extended Controls" = list("", "\\ding{51}", "\\ding{51}", "", "", "\\ding{51}", "\\ding{51}"),
        "Full Controls" = list("", "", "\\ding{51}", "", "", "", "\\ding{51}")
    ),
    custom.note = "\\item \\textit{Notes:} I regress the hospitalization rate on the instrumented weighted upstream deforestation rate. Controls are added consecutively in groups. Climate Controls include precipitation and temperature. Extended Controls add GDP per capita, educational scores, a vaccination index for waterborne diseases. Full controls are the number of doctors per 1000 inhabitants and primary health care coverage (with a reduced panel starting in 2010). Models include municipality and year fixed effects. Standard errors are clustered at the municipalitiy and region-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = .7,
    dcolumn = TRUE
)

second_stage_deter_ex_pop_d <- list()
second_stage_deter_ex_pop_d[["OLS-B"]] <- felm(ex_pop ~ I(1 - forest_upstream_share_d) + temperature + precipitation | municipality + year | 0 | municipality + region_year, data = large_panel)
second_stage_deter_ex_pop_d[["OLS-C"]] <- felm(ex_pop ~ I(1 - forest_upstream_share_d) + temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | 0 | municipality + region_year, data = large_panel)
second_stage_deter_ex_pop_d[["OLS-D"]] <- felm(ex_pop ~ I(1 - forest_upstream_share_d) + temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb + health_primary_care_coverage + health_doctors_1000 | municipality + year | 0 | municipality + region_year, data = small_panel)
second_stage_deter_ex_pop_d[["A"]] <- felm(ex_pop ~ 1 | municipality + year | (I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_upstream) | municipality + region_year, data = large_panel)
second_stage_deter_ex_pop_d[["B"]] <- felm(ex_pop ~ temperature + precipitation | municipality + year | (I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_upstream) | municipality + region_year, data = large_panel)
second_stage_deter_ex_pop_d[["C"]] <- felm(ex_pop ~ temperature + precipitation + gdp_pc + vaccination_index_5y + educ_ideb | municipality + year | (I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_upstream) | municipality + region_year, data = large_panel)
second_stage_deter_ex_pop_d[["D"]] <- felm(ex_pop ~ temperature + precipitation + gdp_pc + educ_ideb + vaccination_index_5y + health_primary_care_coverage + health_doctors_1000 | municipality + year | (I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_upstream) | municipality + region_year, data = small_panel)

texreg(
    second_stage_deter_ex_pop_d,
    file = "output/tables/reg_deforestation_DETER_ex_pop.tex",
    custom.header = list("OLS" = 1:3, "IV" = 4:7), # custom.header = list("Expenditure per Inhabitant" = 1:5),
    custom.model.names = c("B", "C", "D", "A", "B", "C", "D"),
    custom.coef.names = c("Deforestation", "$\\widehat{\\text{Deforestation}}$"),
    omit.coef = c("(temperature)|(precipitation)|(gdp_pc)|(educ_ideb)|(vaccination_index_5y)|(health_primary_care_coverage)|(health_doctors_1000)"),
    stars = c(0.01, 0.05, 0.1),
    digits = 1,
    caption = "IV Regression Results: Upstream Deforestation and Medical Expenditure per Inhabitant",
    label = "tbl-reg-deter-ex-pop-d",
    include.rsquared = FALSE, include.adjrs = FALSE,
    custom.gof.rows = list(
        "Climate Controls" = list("\\ding{51}", "\\ding{51}", "\\ding{51}", "", "\\ding{51}", "\\ding{51}", "\\ding{51}"),
        "Extended Controls" = list("", "\\ding{51}", "\\ding{51}", "", "", "\\ding{51}", "\\ding{51}"),
        "Full Controls" = list("", "", "\\ding{51}", "", "", "", "\\ding{51}")
    ),
    custom.note = "\\item \\textit{Notes:} I regress the medical expenditure per inhabitant on the instrumented weighted upstream deforestation rate. Controls are added consecutively in groups. Climate Controls include precipitation and temperature. Extended Controls add GDP per capita, educational scores, a vaccination index for waterborne diseases. Full controls are the number of doctors per 1000 inhabitants and primary health care coverage (with a reduced panel starting in 2010). Models include municipality and year fixed effects. Standard errors are clustered at the municipalitiy and region-year level. Significance: %stars",
    booktabs = TRUE,
    use.packages = FALSE,
    threeparttable = TRUE,
    scalebox = .7,
    dcolumn = TRUE
)

###
# EXPERIMENTS
###

colnames(large_panel)

test <- large_panel %>%
    group_by(municipality) %>%
    mutate(
        deforestation_rate = (forest - lag(forest, 1)) / total,
        d_pasture_rate = (pasture - lag(pasture, 1)) / total,
        d_agriculture_rate = (agriculture - lag(agriculture, 1)) / total,
        d_urban_rate = (urban - lag(urban, 1)) / total,
        d_mining_rate = (mining - lag(mining, 1)) / total
    ) %>%
    # mutate(deforestation_rate = deforestation / total) %>%
    ungroup() %>%
    filter(year %in% 2005:2020, municipality %/% 1e5 == 5) # %>%
# select(municipality, year, forest, deforestation_rate, cloud_cover_DETER, region_year)

large_panel %>%
    filter(cloud_cover_DETER_upstream == 0) %>%
    select(municipality, year, cloud_cover_DETER, cloud_cover_DETER_upstream)

ggplot(large_panel %>% filter(year < 2018), aes(x = I(1 - forest_d_upstream / total_upstream), y = mortality_rate_tot)) +
    geom_point() +
    geom_smooth(method = "lm")

ggplot(large_panel, aes(x = I(1 - forest_d_upstream / total_upstream), y = mortality_rate_tot, color = year)) +
    geom_point()

felm(I(1 - forest_d_upstream / total_upstream) ~ cloud_cover_DETER_upstream + precipitation + temperature | municipality + year | 0 | municipality + region_year, data = large_panel) %>% summary()


felm(I(1 - forest_d_upstream / total_upstream) ~ cloud_cover_DETER_upstream + precipitation + temperature | municipality + year | 0 | municipality + region_year, data = large_panel) %>% summary()

felm(hosp_rate ~ 1 | municipality + year | (I(1 - forest_d_upstream / total_upstream) ~ cloud_cover_DETER_upstream) | municipality + region_year, data = large_panel_l1) %>% summary()
I(forest_d_upstream / total_upstream)

large_panel %>% select(municipality, year, I(1 - forest_d_upstream / total_upstream), mortality_rate_tot)


inner_join(municipalities, large_panel %>% reframe(CC_2r = as.character(municipality), region = str_sub(region_year, 1, 6)) %>% distinct()) %>%
    ggplot() +
    geom_sf(aes(fill = region))


analysis_test <- analysis %>%
    mutate(legal_amazon = (as.character(CC_2r) %in% legal_amazon), post2005 = year > 2004) %>%
    arrange(CC_2r, year) %>%
    group_by(CC_2r) %>%
    mutate(
        deforestation_rate = 1 - (forest - lag(forest, 1)) / total,
        d_pasture_rate = (pasture - lag(pasture, 1)) / total,
        d_agriculture_rate = (agriculture - lag(agriculture, 1)) / total,
        d_urban_rate = (urban - lag(urban, 1)) / total,
        d_mining_rate = (mining - lag(mining, 1)) / total
    )

felm(deforestation_rate ~ legal_amazon * post2005 | 0 | 0 | CC_2r + year, data = analysis_test) %>% summary()

test_model <- lm(deforestation_rate ~ cloud_cover * legal_amazon * post2005, data = analysis_test)

predict(test_model, newdata = data.frame(cloud_cover = 1, legal_amazon = F, post2005 = T))


felm(I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_upstream + I(1 - forest_upstream_share_d_l1) + I(1 - forest_upstream_share_d_l2) | municipality + year | 0 | municipality + region_year, data = large_panel) %>% summary()

felm(hosp_rate ~ 1 | municipality + year | (I(1 - forest_upstream_share_d) ~ cloud_cover_DETER_upstream + I(1 - forest_upstream_share_d_l1) + I(1 - forest_upstream_share_d_l2)) | municipality + region_year, data = large_panel) %>% summary()
