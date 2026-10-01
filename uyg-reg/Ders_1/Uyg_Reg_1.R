## install.packages("faraway") nolint
## install.packages("scatterplot3d") nolint
install.packages("plotly")

library(scatterplot3d)
library(faraway)
library(plotly)

data(stackloss)
stackloss
help(stackloss)
dim(stackloss)
summary(stackloss)
stackloss$Air.Flow
mean(stackloss$Air.Flow)
median(stackloss$Ai)
range(stackloss$Ai)
quantile(stackloss$Ai)
var(stackloss$Ai)
sqrt(var(stackloss$Ai))

sd <- function(x) sqrt(var(x))
sd(stackloss$Ai)

cor(stackloss)
stem(stackloss$Ai)

hist(stackloss$Ai)
hist(stackloss$Ai, main = "Histogram of Air Flow", xlab = "Flow of cooling air")
boxplot(stackloss$Ai)

plot(stackloss$Ai, stackloss$W)
plot(Water.Temp ~ Air.Flow, stackloss, xlab = "Air Flow", ylab = "Water Temp")

plot(stackloss)

par(mfrow = c(2, 2))
boxplot(stackloss$Ai)
boxplot(stackloss$Wa)
boxplot(stackloss$Ac)
boxplot(stackloss$s)
par(mfrow = c(1, 1))

stackloss[2, ]
stackloss[, 3]
stackloss[2, 3]
c(1, 2, 4)
stackloss[c(1, 2, 4), ]
3:11
stackloss[3:6, ]
stackloss[, -c(1, 2)]
stackloss[stackloss$Ai > 72, ]

data(pima)
pima
summary(pima)

pima$diastolic[pima$diastolic == 0] <- NA
pima$glucose[pima$glucose == 0] <- NA
pima$triceps[pima$triceps == 0] <- NA
pima$insulin[pima$insulin == 0] <- NA
pima$bmi[pima$bmi == 0] <- NA

pima$test <- factor(pima$test)
summary(pima$test)
levels(pima$test) <- c("negative", "positive")
summary(pima)

hist(pima$diastolic)
plot(density(pima$diastolic, na.rm = TRUE))
plot(sort(pima$diastolic), pch = ".")
plot(diabetes ~ diastolic, pima)
plot(diabetes ~ test, pima)
pairs(pima)

data(stat500)
stat500 <- data.frame(scale(stat500))
mean(stat500$midterm)
plot(final ~ midterm, stat500)
#This function adds one or more straight lines through the current plot.default
#(a, b):	the intercept and slope, single values.
abline(0, 1)

g <- lm(final ~ midterm, stat500)
gxx <- lm(final ~ midterm + hw, stat500)
##nolint z_vals <- function(x, y) {
##nolint   gxx$coef[1] + gxx$coef[2] * x + gxx$coef[3] * y
##nolint }
##nolint persp(-10:10, -10:10, outer(-10:10, -10:10, z_vals))
s3d <- scatterplot3d(stat500[, 1:3], pch = 16, color = "#56B4E9", angle = 80)
s3d$plane3d(gxx)
plot_ly(
  stat500,
  x = ~ stat500[, 1], y = ~stat500[, 2], z = stat500[, 3],
  color = "#8AE4E9",
  colors = c("#3a86d4", "#E69F00", "#56B4E9"),
  type = "scatter3d", mode = "markers",
  marker = list(size = 4)
) |>
  layout(scene = list(
    xaxis = list(title = "midterm"),
    yaxis = list(title = "final"),
    zaxis = list(title = "hw")
  ))

abline(g$coef, lty = 5)
cor(stat500)

data(gala)
gala
gfit <- lm(Species ~ Area + Elevation + Nearest + Scruz + Adjacent, data = gala)
summary(gfit)

x <- cbind(1, gala[, -c(1, 2)])
y <- gala$Species

## t(x) %*% x #Error in t(x) %*% x : requires numeric/complex matrix/vector args #nolint

x <- as.matrix(x)
t(x) %*% x
xtxi <- solve(t(x) %*% x)
xtxi

gfit <- lm(Species ~ Area + Elevation + Nearest + Scruz + Adjacent, data = gala)
gs <- summary(gfit)
gs$cov.unscaled

names(gs)
names(gfit)

gfit$fit
gfit$res

xtxi %*% t(x) %*% y
solve(t(x) %*% x, t(x) %*% y)
sqrt(sum(gfit$res^2) / (30 - 6))
sqrt(diag(xtxi)) * 60.975
1 - sum(gfit$res^2) / sum((y - mean(y))^2)
