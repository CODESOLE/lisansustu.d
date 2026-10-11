rm(list = ls())

# install.packages("faraway") nolint
library(faraway)

data(savings)
savings

g <- lm(sr ~ pop15 + pop75 + dpi + ddpi, data = savings)
summary(g)

sum((savings$sr - mean(savings$sr))^2)
sum(g$res^2)

((983.63 - 650.71) / 4) / (650.706 / 45) # F= ((SYY-SSE)/ k)) /(SSE/n-k-1) or

sum((g$fitted.values - mean(savings$sr))^2)
sum(g$residuals^2)
((332.9153) / 4) / (650.706 / 45) # F= Mean square of Regression (MSR) / Mean Square of Error(MSE)

1 - pf(5.755744, 4, 45) # p degeri

g2 <- lm(sr ~ pop75 + dpi + ddpi, data = savings)
sum(g2$res^2)
(797.72 - 650.71) / (650.71 / 45)
1 - pf(10.167, 1, 45)

sqrt(10.167)
2 * (1 - pt(3.1886, 45))

anova(g2, g)

g <- lm(sr ~ ., savings)
gr <- lm(sr ~ I(pop15 + pop75) + dpi + ddpi, savings)
anova(gr, g)

gr <- lm(sr ~ pop15 + pop75 + dpi + offset(ddpi), savings)
anova(gr, g)

tstat <- (0.409695 - 1) / 0.196197
tstat
2 * pt(tstat, 45)
tstat^2

g <- lm(sr ~ pop75 + dpi, data = savings)
summary(g)
gs <- summary(g)
gs$fstat
fstats <- numeric(1000)
for (i in 1:1000) {
  ge <- lm(sample(sr) ~ pop75 + dpi, data = savings)
  fstats[i] <- summary(ge)$fstat[1]
}
length(fstats[fstats > 2.6796]) / 1000

g <- lm(sr ~ ., savings)
summary(g)
qt(0.975, 45)
c(-1.69 - 2.01 * 1.08, -1.69 + 2.01 * 1.08)
c(0.41 - 2.01 * 0.196, 0.41 + 2.01 * 0.196)

install.packages("ellipse")
library(ellipse)
plot(ellipse(g, c(2, 3)), type = "l", xlim = c(-1, 0))
points(0, 0)
points(g$coef[2], g$coef[3], pch = 18)
abline(v = c(-0.461 - 2.01 * 0.145, -0.461 + 2.01 * 0.145), lty = 2)
abline(h = c(-1.69 - 2.01 * 1.08, -1.69 + 2.01 * 1.08), lty = 2)
cor(savings$pop15, savings$pop75)

data(gala)
gala
g <- lm(Species ~ Area + Elevation + Nearest + Scruz + Adjacent, data = gala)
summary(g)
x0 <- c(1, 0.08, 93, 6.0, 12.0, 0.34)
y0 <- sum(x0 * g$coef)
y0
x <- cbind(1, gala[, 3:7])
x <- as.matrix(x)
xtxi <- solve(t(x) %*% x)
bm <- sqrt(x0 %*% xtxi %*% x0) * 2.064 * 60.98
bm
c(y0 - bm, y0 + bm)
bm <- sqrt(1 + x0 %*% xtxi %*% x0) * 2.064 * 60.98
c(y0 - bm, y0 + bm)
predict(g, data.frame(
  Area = 0.08, Elevation = 93, Nearest = 6.0, Scruz = 12,
  Adjacent = 0.34
), se = T)

data(odor)
odor
help(odor)

x <- as.matrix(cbind(1, odor[, -1]))
x
t(x) %*% x
g <- lm(odor ~ temp + gas + pack, data = odor)
summary(g, cor = T)
g <- lm(odor ~ gas + pack, data = odor)
summary(g)
