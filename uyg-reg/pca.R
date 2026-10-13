arr <- matrix(c(
  126, 38,
  128, 40,
  128, 42,
  130, 42,
  130, 44,
  132, 46
), ncol = 2, byrow = TRUE)
# prcomp centers the columns and uses sample variance, matching sklearn PCA defaults # nolint
pca <- prcomp(arr, center = TRUE, scale. = FALSE)
# The first principal component scores (equivalent to arr_pca)
arr_pca <- pca$x[, 1, drop = FALSE]
# sklearn stores components as rows; prcomp stores them as columns
components <- t(pca$rotation[, 1, drop = FALSE])
explained_variance <- pca$sdev[1]^2
explained_variance_ratio <- explained_variance / sum(pca$sdev^2)
plot(pca$sdev, type = "h")
grid()
cat("pca.components_.shape ", paste(dim(components), collapse = " "), "\n")
cat("pca.components_ ", components, "\n")
cat("pca.explained_variance_ ", explained_variance, "\n")
cat("pca.explained_variance_ratio_ ", explained_variance_ratio, "\n")
