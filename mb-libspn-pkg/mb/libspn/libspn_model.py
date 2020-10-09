# Copyright (c) 2020 Julien Klaus (Julien.Klaus@uni-jena.de)
import libspn as spn
import tensorflow as tf

if __name__ == '__main__':
    indicator_leaves = spn.IndicatorLeaf(
        num_vars=2, num_vals=2, name="indicator_x")

    # Connect first two sums to indicators of first variable
    sum_11 = spn.Sum((indicator_leaves, [0, 1]), name="sum_11")
    sum_12 = spn.Sum((indicator_leaves, [0, 1]), name="sum_12")

    # Connect another two sums to indicators of the second variable
    sum_21 = spn.Sum((indicator_leaves, [2, 3]), name="sum_21")
    sum_22 = spn.Sum((indicator_leaves, [2, 3]), name="sum_22")

    # Connect three product nodes
    prod_1 = spn.Product(sum_11, sum_21, name="prod_1")
    prod_2 = spn.Product(sum_11, sum_22, name="prod_2")
    prod_3 = spn.Product(sum_12, sum_22, name="prod_3")

    # Connect a root sum
    root = spn.Sum(prod_1, prod_2, prod_3, name="root")

    # Connect a latent indicator
    indicator_y = root.generate_latent_indicators(name="indicator_y")  # Can be added manually

    # Generate weights
    spn.generate_weights(root, initializer=tf.initializers.random_uniform())  # Can be added manually