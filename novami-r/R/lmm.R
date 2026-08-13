library(lme4)
library(lmerTest)
library(car)
library(performance)
library(emmeans)
library(dplyr)

emm_options(pbkrtest.limit = 20000, lmerTest.limit = 20000, lmer.df = "satterthwaite")

#' Build and validate a random-intercept LMM formula from a pre-filtered data frame.
#'
#' @param data           A data frame.
#' @param response       Character. Name of the numeric response column.
#' @param fixed_effects  Character vector of candidate fixed-effect column names.
#' @param random_effects Character vector of candidate random-effect grouping column names.
#'
#' @return A named list with elements \code{data} (relevelled data frame) and \code{form} (validated formula).
#' @export
get_formula <- function(
    data,
    response,
    fixed_effects  = c(),
    random_effects = c()) {
  
  message("Checking levels of fixed and random effects")
  
  has_levels <- function(v) {
    v %in% names(data) &&
      is.factor(data[[v]]) &&
      nlevels(droplevels(data[[v]])) >= 2
  }
  
  fixed_terms   <- Filter(has_levels, fixed_effects)
  fixed_dropped <- setdiff(fixed_effects, fixed_terms)
  
  random_terms   <- Filter(has_levels, random_effects)
  random_dropped <- setdiff(random_effects, random_terms)
  
  if (length(fixed_dropped) > 0)
    message("Dropping fixed effect(s) with < 2 levels: ",
            paste(fixed_dropped, collapse = ", "))
  
  if (length(random_dropped) > 0)
    message("Dropping random effect(s) with < 2 levels: ",
            paste(random_dropped, collapse = ", "))
  
  if (length(fixed_terms) == 0)
    stop("No usable fixed effects with >= 2 levels.")
  
  if (length(random_terms) == 0)
    stop("No usable random-effect grouping factors with >= 2 levels.")
  
  fixed_part  <- paste(fixed_terms, collapse = " + ")
  random_part <- paste(sprintf("(1 | %s)", random_terms), collapse = " + ")
  form        <- as.formula(paste(response, "~", fixed_part, "+", random_part))
  
  invisible(tryCatch(
    lFormula(formula = form, data = data),
    error = function(e) stop("Formula failed lme4 validation: ", conditionMessage(e))
  ))
  
  list(data = droplevels(data), form = form)
}


#' Fit a REML linear mixed-effects model using the bobyqa optimiser.
#'
#' @param data A data frame, typically the \code{$data} element from \code{\link{get_formula}}.
#' @param form A formula object, typically the \code{$form} element from \code{\link{get_formula}}.
#'
#' @return A named list with elements \code{model}, \code{data}, \code{form},
#'   \code{is_singular}, \code{convergence_warning}, \code{n_fixed_coefs},
#'   and \code{n_random_groups}.
#' @export
fit_lmm <- function(data, form) {
  
  convergence_warning <- NA
  message("Fitting the model")
  
  fit <- withCallingHandlers(
    tryCatch(
      lmerTest::lmer(
        formula = form,
        data    = data,
        REML    = TRUE,
        control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
      ),
      error = function(e) stop("Model fitting failed: ", conditionMessage(e))
    ),
    warning = function(w) {
      convergence_warning <<- conditionMessage(w)
      invokeRestart("muffleWarning")
    }
  )
  
  is_sing <- isSingular(fit)
  if (is_sing) message("Singular fit detected")
  
  list(
    model               = fit,
    data                = data,
    form                = form,
    is_singular         = is_sing,
    convergence_warning = convergence_warning,
    n_fixed_coefs       = length(fixef(fit)),
    n_random_groups     = sapply(ranef(fit), nrow)
  )
}


#' Run a standard post-fit diagnostic battery on a fitted LMM.
#'
#' @param model A fitted \code{lmerMod} or \code{lmerModLmerTest} object,
#'   typically the \code{$model} element from \code{\link{fit_lmm}}.
#'
#' @return A named list with elements \code{anova_table}, \code{vif}, \code{r2},
#'   \code{icc}, \code{varcorr}, \code{coefficients}, and \code{emmeans}
#'   (pairwise contrasts for all terms significant at p < 0.05).
#' @export
analyze_lmm <- function(model) {
  
  message("Running ANOVA")
  anova_table <- tryCatch(
    as.data.frame(stats::anova(model, type = 2, ddf = "Satterthwaite")),
    error = function(e) { message("ANOVA failed: ", conditionMessage(e)); NA }
  )
  
  message("Computing VIF")
  vif <- tryCatch(
    car::vif(model),
    error = function(e) {
      message("VIF failed (likely < 2 fixed-effect terms): ", conditionMessage(e))
      NA
    }
  )
  
  message("Computing R2")
  r2 <- tryCatch(performance::r2(model), error = function(e) NA)
  
  message("Computing ICC")
  icc <- tryCatch(performance::icc(model, by_group = TRUE), error = function(e) NA)
  
  message("Extracting variance components")
  varcorr <- as.data.frame(VarCorr(model))
  
  message("Extracting coefficients")
  coefficients <- summary(model)$coefficients
  
  message("Computing emmeans for significant terms")
  sig_terms <- character(0)
  if (is.data.frame(anova_table) || inherits(anova_table, "anova")) {
    aov_df    <- as.data.frame(anova_table)
    sig_terms <- rownames(aov_df)[which(aov_df[["Pr(>F)"]] < 0.05)]
  }
  
  emmeans_results <- setNames(
    lapply(sig_terms, function(term) {
      tryCatch({
        emm <- emmeans::emmeans(model, specs = term)
        list(
          means    = emm,
          pairwise = tryCatch(pairs(emm, adjust = "tukey"), error = function(e) NA)
        )
      }, error = function(e) {
        message("emmeans failed for '", term, "': ", conditionMessage(e))
        NA
      })
    }),
    sig_terms
  )
  
  message("Done.")
  
  list(
    anova_table  = anova_table,
    vif          = vif,
    r2           = r2,
    icc          = icc,
    varcorr      = varcorr,
    coefficients = coefficients,
    emmeans      = emmeans_results
  )
}

#' Pretty-print the results of analyze_lmm.
#'
#' @param results A named list returned by \code{\link{analyze_lmm}}.
#'
#' @return Invisibly returns \code{results} (called for its side effects).
#' @export
print_results <- function(results) {
  message("=== ANOVA ===");        print(results$anova_table)
  message("=== VIF ===");          print(results$vif)
  message("=== R2 ===");           print(results$r2)
  message("=== ICC ===");          print(results$icc)
  message("=== Var. Corr. ===");   print(results$varcorr)
  message("=== Coeff. ===");       print(results$coefficients)
  message("=== EMMEANS ===");      print(results$emmeans)
  invisible(results)
}

