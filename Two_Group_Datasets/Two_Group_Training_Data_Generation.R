options(scipen = 9999)

rm(list = ls())

# Setup ----

S <- 2  # Number of groups (rows)
J <- 10 # Number of items (columns)
dif.mu <- c(0, 1)
sigma <- c(1, 1)
grid.a = seq(-1, 1, by = 0.25)
grid.a = grid.a[grid.a != 0]
grid.b = seq(-1.5, 1.5, by = 0.25)
grid.b = grid.b[grid.b != 0]
ratio <- c(.6, .4)

# Sample Size with higher probability of N < 1000
n_vals <- 100:1000
valid_n <- n_vals[n_vals %% 5 == 0] # Make sure we have integer group sizes
weights <- ifelse(valid_n < 750, 3, 1)  # 3 times more likely if  N < 750
weights <- weights / sum(weights)

for(r in 1:600){ # Number of data sets to generate
  set.seed(202407)
  set.seed(sample.int(2147483647, 10000)[S * r])
  # Random Sample Size for training data
  N  = sample(valid_n, size = 1, prob = weights)
  # Random Number of DIF items
  m = sample(c(2, 3, 4), 1)
  # Group Sizes
  g <- do.call(c, lapply(1:S, function(s) rep(s, N * ratio[s])))
  # Randomly select the m many items to have DIF
  selected_items <- sample(1:J, m, replace = FALSE)
  # Randomly draw for DIF Shift
  dif.a = sample(grid.a, size = S, replace = T)
  dif.a[1] = 0
  dif.b = sample(grid.b, size = S, replace = T)
  dif.b[1] = 0
  
  # Generate item parameters
  dif_types <- c("b", "ab", "a")  
  rep_dif = sample(dif_types, 1)
  
  a0 <- runif(J, 1.5, 2.5)
  b0 <- rnorm(J)
  a <- t(replicate(S, a0))
  b <- t(replicate(S, b0))
  
  # Apply DIF shift
  if(rep_dif == "ab"){
    a[, selected_items] <- a[, selected_items] + dif.a
    b[, selected_items] <- b[, selected_items] + dif.b
  } else if(rep_dif == "b"){
    b[, selected_items] <- b[, selected_items] + dif.b
  } else if(rep_dif == "a"){
    a[, selected_items] <- a[, selected_items] + dif.a
  }
  
  # Storage for DIF Shift
  d_a <- array(0, c(S, S, J))
  d_b <- array(0, c(S, S, J))
  for (k in 1:(S - 1))
    for (n in (k + 1):S) {
      d_a[k, n, ] <- a[k, ] - a[n, ]
      d_b[k, n, ] <- b[k, ] - b[n, ]
    }
  
  # Create target labels
  temp_d_a <- combn(1:nrow(a), 2, function(idx) {
    diff <- a[idx[1], ] - a[idx[2], ]
    return(diff)
  })
  colnames(temp_d_a) = "d.a"
  
  temp_d_b <- combn(1:nrow(a), 2, function(idx) {
    diff <- b[idx[1], ] - b[idx[2], ]
    return(diff)
  })
  colnames(temp_d_b) = "d.b"
  
  # Create Labels for DIF Between Groups
  Labels_a <- temp_d_a
  Labels_a[Labels_a != 0] <- 1
  colnames(Labels_a) = "DIF_a"
  
  Labels_b <- temp_d_b
  Labels_b[Labels_b != 0] <- 1
  colnames(Labels_b) = "DIF_b"
  
  Labels_DIF <- pmax(Labels_a, Labels_b)
  colnames(Labels_DIF) = "DIF"
  
  
  # Generate observed responses
  y <- t(sapply(1:N, function(n) {
    s <- g[n]
    theta <- rnorm(1, dif.mu[s], sigma[s])
    rbinom(J, 1, plogis(a[s, ] * theta - b[s, ]))
  }))
  
  save.image(file = paste0("Two_Group_Training_Data_Replication_", r, ".RData"))
}
