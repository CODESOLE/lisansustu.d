mu <- 4
sigma <- 2

sample <- rnorm(1, mean = mu, sd = sigma)
prev <- dnorm(sample, mu, sigma)
monte <- list(sample)
for (i in 1:999) {
  sample <- rnorm(1, mean = mu, sd = sigma)
  l <- dnorm(sample, mu, sigma)
  if (l < prev) {
      cond <- sample(c(1, 0), size = 1, replace = TRUE, prob = c(l / prev, 1 - l / prev)) # nolint
      if (cond == 1) monte <- c(monte, sample) else next
  } else {
    monte <- c(monte, sample)
  }
  prev <- l
}
mcmc <- unlist(monte, use.names = FALSE)
hist(mcmc, bins = 10)
