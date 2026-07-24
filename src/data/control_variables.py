import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

class control_variables:
    """
    A class to preprocess economic, education, health, vaccination, and sanitation data.
    """
    
    def __init__(self):
        pass
    
    def fetch(self):
        """
        Fetches the required data. Automatic downloads are not supported.
        """
        pass
    
    def preprocess(self):
        """
        Preprocesses the data by performing various operations such as combining datasets,
        creating new variables, cleaning, adjusting magnitudes, and saving processed data.
        """
        
        # ===========================================================
        # GDP Data Preprocessing
        # ===========================================================
        
        # import
        gdp = pd.read_csv('data/misc/raw/gdp.csv')
        gdp_old = pd.read_csv('data/misc/raw/gdp_old.csv')

        ## combine: take old data until 2001 and new data from 2002; link at gdp in 2002
        t_linkage = pd.merge(gdp_old.query("ano == 2002").set_index("id_municipio").pib, gdp.query("ano == 2002").set_index("id_municipio").pib, left_index=True, right_index=True)
        t_linkage = (t_linkage.pib_x / t_linkage.pib_y).to_dict()
        gdp_old["pib_linked"] = gdp_old.apply(lambda x: x.pib * t_linkage[x.id_municipio] if x.id_municipio in t_linkage else np.nan, axis=1)
        gdp_combined = pd.concat([gdp_old.query("ano < 2002"), gdp.query("ano >= 2002")]).reset_index(drop=True)

        # create variables
        gdp_combined["gdp"] = gdp_combined["pib_linked"].fillna(gdp_combined["pib"])
        gdp_combined["gva_share_agriculture"] = gdp_combined["va_agropecuaria"] / gdp_combined["va"]
        gdp_combined["gva_share_industry"] = gdp_combined["va_industria"] / gdp_combined["va"]
        gdp_combined["gva_share_services"] = gdp_combined["va_servicos"] / gdp_combined["va"]
        gdp_combined["gva_share_public"] = gdp_combined["va_adespss"] / gdp_combined["va"]
        gdp_combined["year"] = gdp_combined["ano"]
        gdp_combined["CC_2"] = gdp_combined["id_municipio"]
        gdp_combined["CC_2r"] = gdp_combined["CC_2"].astype(str).str.slice(0, 6).astype(int)

        # clean
        gdp_combined = gdp_combined[["CC_2r", "year", "gdp", "gva_share_agriculture", "gva_share_industry", "gva_share_services", "gva_share_public"]].dropna()

        def sliding_window_linear_regression(series):
            """
            Perform linear regression in sliding windows and return the median coefficients.

            :param series: Time series data.
            :return: Median coefficients of the linear regression.
            """
            n = len(series)
            window_size = n // 2
            coefficients = []

            # Fit linear models in sliding windows
            for start in range(0, n - window_size + 1, 1):
                end = start + window_size
                window_data = series.iloc[start:end]
                X = sm.add_constant(window_data.index.values.reshape(-1, 1))  # Add intercept
                y = window_data.values
                model = sm.OLS(y, X).fit()
                coefficients.append(model.params)

            return np.median(coefficients, axis=0)

        def adjust_magnitudes(series, threshold=1.5):
            """
            Adjust magnitudes of a series to mitigate outliers using linear regression and IQR.

            :param series: Time series data.
            :param threshold: IQR threshold for identifying outliers.
            :return: Adjusted series.
            """
            if len(series) < 2:  # Not enough data to adjust
                return series

            model_coefs = sliding_window_linear_regression(series)
            X_full = sm.add_constant(series.index.values.reshape(-1, 1))  # Add intercept
            predictions = np.dot(X_full, model_coefs)
            residuals = series - predictions

            # Calculate IQR for residuals
            Q1 = np.percentile(residuals, 25)
            Q3 = np.percentile(residuals, 75)
            IQR = Q3 - Q1

            # Identify outliers
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Adjust outliers
            adjusted_series = series.copy()
            for i, residual in enumerate(residuals):
                if residual < lower_bound or residual > upper_bound:
                    adjusted_series.iat[i] = predictions[i]  # Adjust to fitted value

            return adjusted_series
        for col in ["gdp", "gva_share_agriculture", "gva_share_industry", "gva_share_services", "gva_share_public"]:
            gdp_combined[col] = gdp_combined.groupby("CC_2r")[col].transform(adjust_magnitudes)
        gdp_combined.to_parquet('data/misc/raw/gdp_processed.parquet', index=False)

        # ===========================================================
        # Education Data Preprocessing
        # ===========================================================

        education = pd.read_csv("data/misc/raw/education.csv")
        education["CC_2r"] = education["id_municipio"].astype(str).str.slice(0, 6).astype(int)
        education["year"] = education["ano"]
        education = education.groupby(["CC_2r", "year"], as_index = False).agg({"ideb": "mean"}).sort_values(["CC_2r", "year"])
        # Identify the range of years
        min_year = education['year'].min()
        max_year = education['year'].max()

        # Identify all unique IDs
        ids = education['CC_2r'].unique()

        # Create a complete DataFrame with all years for each ID
        all_years = pd.DataFrame({
            'year': range(min_year, max_year + 1)
        })
        all_ids_years = all_years.assign(key=1).merge(pd.DataFrame({'CC_2r': ids, 'key': 1}), on='key').drop('key', axis=1)

        # Merge with the original data
        education_full = pd.merge(all_ids_years, education, on=['CC_2r', 'year'], how='left')

        # Perform linear interpolation
        education_full["educ_ideb"] = education_full.groupby('CC_2r')["ideb"].transform(lambda group: group.interpolate())
        education_full.to_parquet('data/misc/raw/education_processed.parquet', index=False)

        # ===========================================================
        # Health Data Preprocessing
        # ===========================================================
        
        health = pd.read_csv("data/misc/raw/ieps_health.csv")
        health["CC_2r"] = health["id_municipio"].astype(str).str.slice(0, 6).astype(int)
        health["year"] = health["ano"]
        health = health[["CC_2r", "year", "cob_ab", "tx_med_ch"]].copy()
        health.rename(columns={"cob_ab": "health_primary_care_coverage", "tx_med_ch": "health_doctors_1000"}, inplace=True)
        health["health_primary_care_coverage"] = health["health_primary_care_coverage"] / 100
        health.to_parquet('data/misc/raw/health_processed.parquet', index=False)

        # ===========================================================
        # Vaccination Data Preprocessing
        # ===========================================================

        vaccinations = pd.read_csv("data/misc/raw/vaccinations.csv")
        vaccinations["CC_2r"] = vaccinations["id_municipio"].astype(str).str.slice(0, 6).astype(int)
        vaccinations["year"] = vaccinations["ano"]
        vaccinations.sort_values(["CC_2r", "year"], inplace=True)
        # calculate rolling average of 5 years
        for col in ["cobertura_febre_amarela", "cobertura_haemophilus_influenza_b", "cobertura_hepatite_a", "cobertura_hepatite_b", "cobertura_poliomielite"]:
            vaccinations[col + "_5y"] = vaccinations.groupby("CC_2r")[col].transform(lambda x: x.rolling(5, min_periods=1).mean())
        # calculate index of all vaccinations
        vaccinations["vaccination_index"] = vaccinations[["cobertura_febre_amarela", "cobertura_haemophilus_influenza_b", "cobertura_hepatite_a", "cobertura_hepatite_b", "cobertura_poliomielite"]].mean(axis=1) / 100
        vaccinations["vaccination_index_5y"] = vaccinations[["cobertura_febre_amarela_5y", "cobertura_haemophilus_influenza_b_5y", "cobertura_hepatite_a_5y", "cobertura_hepatite_b_5y", "cobertura_poliomielite_5y"]].mean(axis=1) / 100
        vaccinations[['CC_2r', 'year', 'vaccination_index', 'vaccination_index_5y']].to_parquet('data/misc/raw/vaccinations_processed.parquet', index=False)

        # ===========================================================
        # Sanitation Data Preprocessing
        # ===========================================================
        
        sanitation = pd.read_csv("data/misc/raw/sanitation.csv")
        sanitation["CC_2r"] = sanitation["id_municipio"].astype(str).str.slice(0, 6).astype(int)
        sanitation["year"] = sanitation["ano"]
        sanitation = sanitation[["CC_2r", "year", "populacao_urbana", "populacao_urbana_atendida_agua", "populacao_urbana_residente_esgoto"]].copy()
        sanitation.rename(columns={"populacao_urbana": "urban_population", "populacao_urbana_atendida_agua": "urban_population_served_water", "populacao_urbana_residente_esgoto": "population_with_sewage"}, inplace=True)
        sanitation.sort_values(["CC_2r", "year"], inplace=True)
        sanitation.to_parquet('data/misc/raw/sanitation_processed.parquet', index=False)

        # ===========================================================
        # Combine Control Variables
        # ===========================================================
        
        gdp_combined = pd.read_parquet('data/misc/raw/gdp_processed.parquet')
        education = pd.read_parquet('data/misc/raw/education_processed.parquet')
        health = pd.read_parquet('data/misc/raw/health_processed.parquet')
        vaccinations = pd.read_parquet('data/misc/raw/vaccinations_processed.parquet')
        sanitation = pd.read_parquet('data/misc/raw/sanitation_processed.parquet')

        data = pd.merge(gdp_combined, education, on=["CC_2r", "year"], how="outer")
        data = pd.merge(data, health, on=["CC_2r", "year"], how="outer")
        data = pd.merge(data, vaccinations, on=["CC_2r", "year"], how="outer")
        data = pd.merge(data, sanitation, on=["CC_2r", "year"], how="outer")

        data.to_parquet('data/misc/control_variables.parquet', index=False)