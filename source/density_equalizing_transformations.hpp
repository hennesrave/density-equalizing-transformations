#pragma once

void density_equalizing_transformation_integral_images(
    float* points,
    int point_count,
    int resolution,
    int kernel_radius,
    int iterations
);

void density_equalizing_transformation_sector_based(
    float* points,
    int point_count,
    int sector_count,
    int iterations
);

void density_equalizing_transformation_multiresolution(
    float* points,
    int point_count,
    int maximum_resolution,
    int kernel_radius,
    int cycle_count
);