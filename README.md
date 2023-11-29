# Artificial_Bee_Colony

Algorithm Explanation:

there is 2 classes: abc/bee
abc class is were the algorithm starts, with bees
bee class is actually represents each solution that has data and fitness

"employed bees":
1- we make some random valid solutions (bees) (in amount of the of the "population number")
2- for each "bee" we do the "cross-over" and "mutation", one time (if doing the "cross-over" or "mutation" does not make improvement, we only add one to "improvement_try" property of that bee, if it does make improvement, we replace it with the original "bee" and make "improvement_try" property of this "new_bee" to zero )


"onlooker bees":
1- we choose a "bee" by "roulette wheel" procedure (we make sure that, the "improvement_try" property of that bee, be less than the "max_improvement_try" vriable) (we do this in amount of "RW_iteration" variable)
2- we pass the choosen "bee" to "try_for_improvement"
3- in "try_for_improvement" we do "cross-over" and "mutation"
4- in "cross-over" we select a random position and random neighbor (the neighbor must not be equal to itself) - we replace this "new_bee "to the original one, if and only if it be valid and had improvement
5- in "mutation" we select a random position, and random target (0, 1) - we replace this "new_bee "to the original one, if and only if it be valid and had improvement
6- we check that if doing cross-over and mutation had made any change or not, for any bee that change has be made, we add one to that bee's "improvement_try" property
7- if the last section, executes (adding one), we pass that bee to "_scout_check", to check that if the bee has reached to "max_improvement_try" or not
8- if in last section, if the bee has reached to "max_improvement_try", we pass it to "scout_bees"
to make a random valid new bee and add it to our "population"


"scout bees":
1- here we make brand new bees and add it to our population