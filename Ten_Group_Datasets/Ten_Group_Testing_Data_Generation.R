options(scipen = 9999)
rm(list = ls())
# Setup ----
S <- 10  # Number of groups
J <- 10 # Number of items
ratio <- c(.1, .1, .15, .15, .05, .05, .15, .15, .05, .05)
sigma <- c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
dif.a = c(0, 0, 0.5, 0.5, -0.5, -0.5, 1, 1, -1, -1)
dif.b = c(0, 0, 1, -1, 1, -1, 1.5, -1.5, 1.5, -1.5)
dif.mu = c(0, 0, 1, 1, 1, 1, -1, -1, -1, -1)
for (N in c(1000, 2000, 4000)) { # Sample Size
  for (m in c(2, 4, 6)) { # Number of DIF Items
    for(r in 1:250){ # Number of data sets to generate
      # Seed for random numbers
      set.seed(202407)
      set.seed(sample.int(2147483647, 10000)[S * r])
      # Group Sizes
      g <- do.call(c, lapply(1:S, function(s) rep(s, N * ratio[s])))
      
      # Randomly select the m many items to have DIF
      selected_items <- sample(1:J, m, replace = FALSE)
      
      # Generate item parameters
      dif_types <- c("b", "ab")
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
      colnames(temp_d_a) = apply(combn(1:ncol(a), 2), 2,
                                 function(idx) paste0("d.a_", "Group", idx[1],
                                                      "Group", idx[2]))
      
      temp_d_b <- combn(1:nrow(a), 2, function(idx) {
        diff <- b[idx[1], ] - b[idx[2], ]
        return(diff)
      })
      colnames(temp_d_b) = apply(combn(1:ncol(b), 2), 2,
                                 function(idx) paste0("d.b_", "Group",
                                                      idx[1], "Group", idx[2]))
      
      # Create Labels for DIF Between Groups
      Labels_a <- temp_d_a
      Labels_a[Labels_a != 0] <- 1
      colnames(Labels_a) = apply(combn(1:ncol(a), 2), 2, 
                                 function(idx) paste0("DIF_a_", "Group", 
                                                      idx[1], "Group", idx[2]))
      
      Labels_b <- temp_d_b
      Labels_b[Labels_b != 0] <- 1
      colnames(Labels_b) = apply(combn(1:ncol(b), 2), 2, 
                                 function(idx) paste0("DIF_b_", "Group", 
                                                      idx[1], "Group", idx[2]))
      
      Labels_DIF <- pmax(Labels_a, Labels_b)
      colnames(Labels_DIF) = apply(combn(1:ncol(b), 2), 2, 
                                   function(idx) paste0("DIF_", "Group", 
                                                        idx[1], "Group", idx[2]))
      
      # Generate theta values and observed responses
      theta <- sapply(1:N, function(n) {
        s <- g[n]
        rnorm(1, dif.mu[s], sigma[s])
      })
      
      y <- t(sapply(1:N, function(n) {
        s <- g[n]
        rbinom(J, 1, plogis(a[s, ] * theta[n] - b[s, ]))
      }))
      
      save.image(file = paste0("Updated_Ten_Group_Testing_Data_", N, "_", m*10, "_Replication_", r, ".RData"))
    }
  }
}