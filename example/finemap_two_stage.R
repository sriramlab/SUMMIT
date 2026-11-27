# Two-stage SuSiE fine-mapping with a SUMMIT-MIX regional matrix (susieR). The error of a composed matrix has the shape
# of the matrix itself, so the first fit (residual variance 1) gives the posterior second moments of the effects, and
# the second fit uses the residual variance s^2 = 1 + mean_j sum_c V_jc E(b_c^2), V the matrix's predicted error
# variance. The model is z = R b + e with e ~ N(0, s^2 (R + lambda I)), lambda = 1e-4, fitted on the z scale through
# susie_suff_stat after whitening by R + lambda I.
#
# Make the inputs with
#   summit admixed-ld matrix --ref REF --params PREFIX.admixed_ld.json --curve CURVE --region CHR:START-END \
#     --sumstats GWAS.sumstats --error-variance --out REGION
# which writes REGION.ld (R), REGION.ld_var (V) and REGION.z.tsv (z on the matrix's counted allele, with N).
# usage: Rscript finemap_two_stage.R REGION [L=10]
# Writes REGION.posterior1.tsv (SNP, EB2: the first fit's E(b^2) on the z scale, the input of
# `summit admixed-ld finemap-variance`), REGION.pip.tsv (SNP, PIP of both fits) and REGION.cs.tsv (95% credible sets
# of the second fit with purity at least 0.5).
suppressMessages(library(susieR))
a <- commandArgs(TRUE)
if (length(a) < 1) stop("usage: Rscript finemap_two_stage.R REGION [L]")
reg <- a[1]
L <- if (length(a) >= 2) as.integer(a[2]) else 10L
LAMBDA <- 1e-4

# unname: read.table names the columns V1, V2, ... and not the rows, and a matrix whose row and column names differ
# fails isSymmetric(), which susie_get_cs checks (the values are written exactly symmetric)
R <- unname(as.matrix(read.table(paste0(reg, ".ld"))))
V <- unname(as.matrix(read.table(paste0(reg, ".ld_var"))))
zt <- read.table(paste0(reg, ".z.tsv"), header = TRUE, sep = "\t", stringsAsFactors = FALSE)
p <- nrow(R)
stopifnot(ncol(R) == p, all(dim(V) == c(p, p)), nrow(zt) == p)
V <- (V + t(V)) / 2
diag(V) <- 0
n <- median(zt$N)
z <- zt$Z * sqrt((n - 1) / (zt$Z^2 + n - 2))            # the z-scores as correlations' scale, as susie_rss does with n

e <- eigen(R, symmetric = TRUE)
U <- e$vectors
ev <- pmax(e$values, 0)
fitl <- function(z, lam, rv) {
  XtX <- U %*% (t(U) * (ev^2 / (ev + lam)))
  uz <- as.vector(crossprod(U, z))
  Xty <- as.vector(U %*% (uz * ev / (ev + lam)))
  yty <- sum(uz^2 / (ev + lam))
  suppressWarnings(susie_suff_stat(XtX = XtX, Xty = Xty, yty = yty, n = p, L = L, residual_variance = rv,
                                   estimate_residual_variance = FALSE, check_input = FALSE, refine = FALSE))
}

f1 <- fitl(z, LAMBDA, 1)                                 # stage 1
eb2 <- colSums(f1$alpha * f1$mu2) / f1$X_column_scale_factors^2
write.table(data.frame(SNP = zt$SNP, EB2 = eb2), paste0(reg, ".posterior1.tsv"), sep = "\t", quote = FALSE,
            row.names = FALSE)
s2 <- 1 + mean(V %*% eb2)                                # the same as `summit admixed-ld finemap-variance`
f2 <- fitl(z, LAMBDA, s2)                                # stage 2

write.table(data.frame(SNP = zt$SNP, PIP_stage1 = f1$pip, PIP = f2$pip), paste0(reg, ".pip.tsv"), sep = "\t",
            quote = FALSE, row.names = FALSE)
cs <- susie_get_cs(f2, Xcorr = cov2cor(R), coverage = 0.95, min_abs_corr = 0.5)
rows <- if (length(cs$cs) == 0) data.frame(cs = integer(0), snps = character(0), coverage = numeric(0),
                                            purity = numeric(0)) else
  do.call(rbind, lapply(seq_along(cs$cs), function(i) data.frame(cs = i, snps = paste(zt$SNP[cs$cs[[i]]], collapse = ";"),
                                                               coverage = cs$coverage[i],
                                                               purity = cs$purity$min.abs.corr[i])))
write.table(rows, paste0(reg, ".cs.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("%s: %d SNPs, N %.0f, s^2 %.5f; %d credible set(s); wrote %s.posterior1.tsv, .pip.tsv, .cs.tsv\n",
            reg, p, n, s2, length(cs$cs), reg))
