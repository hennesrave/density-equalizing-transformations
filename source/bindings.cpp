#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "density_equalizing_transformations.hpp"

NB_MODULE( _density_equalizing_transformations, m )
{
    using points_t = nanobind::ndarray<float, nanobind::shape<-1, 2>, nanobind::c_contig>;

    m.def( "density_equalizing_transformation_integral_images", [] ( points_t points, int resolution, int kernel_radius, int iterations )
    {
        density_equalizing_transformation_integral_images(
            points.data(),
            static_cast<int>( points.shape( 0 ) ),
            resolution,
            kernel_radius,
            iterations
        );
    },
        nanobind::arg { "points" }.noconvert(),
        nanobind::arg { "resolution" } = 1024,
        nanobind::arg { "kernel_radius" } = 8,
        nanobind::arg { "iterations" } = 50
    );

    m.def( "density_equalizing_transformation_sector_based", [] ( points_t points, int sector_count, int iterations )
    {
        density_equalizing_transformation_sector_based(
            points.data(),
            static_cast<int>( points.shape( 0 ) ),
            sector_count,
            iterations
        );
    },
        nanobind::arg { "points" }.noconvert(),
        nanobind::arg { "sector_count" } = 32,
        nanobind::arg { "iterations" } = 50
    );

    m.def( "density_equalizing_transformation_multiresolution", [] ( points_t points, int maximum_resolution, int kernel_radius, int cycle_count )
    {
        density_equalizing_transformation_multiresolution(
            points.data(),
            static_cast<int>( points.shape( 0 ) ),
            maximum_resolution,
            kernel_radius,
            cycle_count
        );
    },
        nanobind::arg { "points" }.noconvert(),
        nanobind::arg { "maximum_resolution" } = 1024,
        nanobind::arg { "kernel_radius" } = 2,
        nanobind::arg { "cycle_count" } = 50
    );
}