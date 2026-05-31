# Define the custom extract function for splm objects
extract.splm <- function(model, include.coefficients = TRUE, include.se = TRUE, 
                         include.pvalues = TRUE, include.rsquared = TRUE, 
                         include.adjrs = TRUE, include.nobs = TRUE, ...) {
  
  # Extract coefficients and their standard errors
  coefficients <- model$coefficients
  se <- sqrt(diag(model$vcov))
  
  # Calculate p-values
  pvalues <- 2 * pt(-abs(coefficients / se), df = nrow(model$model) - length(coefficients))
  
  # Extract other model statistics
  logLik_value <- model$logLik
  nobs <- nrow(model$model)
  
  # Create a texreg object
  tr <- createTexreg(
    coef.names = names(coefficients),
    coef = coefficients,
    se = se,
    pvalues = pvalues,
    gof.names = c("Log Likelihood", "Num. obs."),
    gof = c(logLik_value, nobs)
  )
  
  return(tr)
}
