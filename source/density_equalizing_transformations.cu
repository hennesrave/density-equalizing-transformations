#include "density_equalizing_transformations.hpp"

#include <cuda_runtime.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <iostream>
#include <thread>
#include <vector>

// ----- Constants ----- //

namespace constants
{
    static constexpr int maximum_resolution     = 1024;
    static constexpr float pi                   = 3.14159265358979323846f;
    static constexpr float two_pi               = 2.0f * pi;
}

// ----- Utility ----- //

namespace utility
{
    void assert_error( cudaError_t error, const char* message )
    {
        if( error != cudaSuccess )
        {
            std::cerr << "[Error] " << message << ": " << cudaGetErrorString( error ) << std::endl;
            throw std::runtime_error( message );
        }
    }

    float* initialize_kernel_weights_device( int kernel_radius )
    {
        const auto kernel_sigma         = kernel_radius / 6.0f;

        auto kernel_weights             = std::vector<float>( kernel_radius + 1 );
        auto kernel_weights_total       = kernel_weights[0] = 1.0f;
        for( auto delta = 1; delta <= kernel_radius; ++delta )
        {
            const auto weight       = std::exp( -( delta * delta ) / ( 2.0f * kernel_sigma * kernel_sigma ) );
            kernel_weights[delta]   = weight;
            kernel_weights_total    += 2.0f * weight;
        }
        for( auto& weight : kernel_weights )
        {
            weight /= kernel_weights_total;
        }

        float* kernel_weights_device    = nullptr;
        const auto kernel_weights_bytes = ( kernel_radius + 1 ) * sizeof( float );
        utility::assert_error( cudaMalloc( &kernel_weights_device, kernel_weights_bytes ), "Failed to allocate device memory for kernel weights" );
        utility::assert_error( cudaMemcpy( kernel_weights_device, kernel_weights.data(), kernel_weights_bytes, cudaMemcpyHostToDevice ), "Failed to copy kernel weights to device" );

        return kernel_weights_device;
    }

    __device__ __forceinline__ int reflect_index( int index, int maximum_index )
    {
        return
            ( index < 0 )               ? -( index + 1 ) :
            ( index > maximum_index )   ? 2 * maximum_index - index + 1 :
            index;
    }

    __device__ float load( const float* texture, int2 coordinates, int dimensions )
    {
        return texture[coordinates.y * dimensions + coordinates.x];
    }
    __device__ float load( const float* texture, int2 coordinates, int dimensions, float default_value )
    {
        if( coordinates.x < 0 || coordinates.x >= dimensions || coordinates.y < 0 || coordinates.y >= dimensions )
        {
            return default_value;
        }
        return load( texture, coordinates, dimensions );
    }

    __device__ float2 load( const float2* texture, int2 coordinates, int dimensions )
    {
        return texture[coordinates.y * dimensions + coordinates.x];
    }
    __device__ float2 load( const float2* texture, int2 coordinates, int dimensions, float2 default_value )
    {
        if( coordinates.x < 0 || coordinates.x >= dimensions || coordinates.y < 0 || coordinates.y >= dimensions )
        {
            return default_value;
        }
        return load( texture, coordinates, dimensions );
    }
}

// ----- Kernels ----- //

namespace kernels
{
    __global__ void fill( float* values, int value_count, float value )
    {
        for( int index = blockIdx.x * blockDim.x + threadIdx.x; index < value_count; index += blockDim.x * gridDim.x )
        {
            values[index] = value;
        }
    }

    __global__ void compute_density_global_atomic( const float2* points, int point_count, float* density, int resolution )
    {
        const auto point_index = blockIdx.x * blockDim.x + threadIdx.x;
        if( point_index < point_count )
        {
            const auto point = points[point_index];
            const auto x = min( __float2int_rz( point.x * resolution ), resolution - 1 );
            const auto y = min( __float2int_rz( point.y * resolution ), resolution - 1 );

            atomicAdd( density + y * resolution + x, 1 );
        }
    }
    __global__ void compute_density_shared_atomic( const float2* points, int point_count, float* density, int resolution )
    {
        extern __shared__ float shared_density[];
        for( int density_index = threadIdx.x; density_index < resolution * resolution; density_index += blockDim.x )
        {
            shared_density[density_index] = 0.0f;
        }
        __syncthreads();

        const auto point_index = blockIdx.x * blockDim.x + threadIdx.x;
        if( point_index < point_count )
        {
            const auto point = points[point_index];
            const auto x = min( __float2int_rz( point.x * resolution ), resolution - 1 );
            const auto y = min( __float2int_rz( point.y * resolution ), resolution - 1 );

            atomicAdd( shared_density + y * resolution + x, 1.0f );
        }
        __syncthreads();

        for( int density_index = threadIdx.x; density_index < resolution * resolution; density_index += blockDim.x )
        {
            const auto y = density_index / resolution;
            const auto x = density_index % resolution;

            atomicAdd( density + y * resolution + x, shared_density[density_index] );
        }
    }

    __global__ void smooth_density_horizontal( float* density, int resolution, const float* kernel_weights, int kernel_radius )
    {
        extern __shared__ float shared_density[];

        const auto x = static_cast<int>( threadIdx.x );
        const auto y = static_cast<int>( blockIdx.x );

        for( auto index = static_cast<int>( threadIdx.x ); index < blockDim.x + 2 * kernel_radius; index += blockDim.x )
        {
            const auto density_x = utility::reflect_index( index - kernel_radius, resolution - 1 );
            shared_density[index] = density[y * resolution + density_x];
        }
        __syncthreads();

        auto smoothed_density = kernel_weights[0] * shared_density[x + kernel_radius];
        for( auto delta = 1; delta <= kernel_radius; ++delta )
        {
            smoothed_density += kernel_weights[delta] * shared_density[x + kernel_radius + delta];
            smoothed_density += kernel_weights[delta] * shared_density[x + kernel_radius - delta];
        }

        density[y * resolution + x] = static_cast<float>( smoothed_density );
    }
    __global__ void smooth_density_vertical( float* density, int resolution, const float* kernel_weights, int kernel_radius )
    {
        extern __shared__ float shared_density[];

        const auto x = static_cast<int>( blockIdx.x );
        const auto y = static_cast<int>( threadIdx.x );

        for( auto index = static_cast<int>( threadIdx.x ); index < blockDim.x + 2 * kernel_radius; index += blockDim.x )
        {
            const auto density_y = utility::reflect_index( index - kernel_radius, resolution - 1 );
            shared_density[index] = density[density_y * resolution + x];
        }
        __syncthreads();

        auto smoothed_density = kernel_weights[0] * shared_density[y + kernel_radius];
        for( auto delta = 1; delta <= kernel_radius; ++delta )
        {
            smoothed_density += kernel_weights[delta] * shared_density[y + kernel_radius + delta];
            smoothed_density += kernel_weights[delta] * shared_density[y + kernel_radius - delta];
        }

        density[y * resolution + x] = static_cast<float>( smoothed_density );
    }

    __global__ void compute_integral_columns( const float* density, float2* integral_columns )
    {
        __shared__ float2 values[constants::maximum_resolution];

        const auto pixel_index      = threadIdx.x * blockDim.x + blockIdx.x;
        const auto value_index      = static_cast<int>( threadIdx.x );

        const auto pixel_density    = density[pixel_index];
        values[value_index]         = make_float2( pixel_density, pixel_density );

        for( auto stepsize = 1; stepsize < blockDim.x; stepsize *= 2 )
        {
            __syncthreads();
            const auto x = ( value_index - stepsize >= 0 ) ? values[value_index - stepsize].x : 0.0f;
            const auto y = ( value_index + stepsize < blockDim.x ) ? values[value_index + stepsize].y : 0.0f;

            __syncthreads();
            values[value_index].x += x;
            values[value_index].y += y;
        }

        integral_columns[pixel_index] = values[value_index];
    }
    __global__ void compute_integral_image( const float2* integral_columns, float* integral_image )
    {
        __shared__ float values[constants::maximum_resolution];

        const auto pixel_index  = blockIdx.x * blockDim.x + threadIdx.x;
        const auto value_index  = static_cast<int>( threadIdx.x );
        values[value_index]     = integral_columns[pixel_index].x;

        for( auto stepsize = 1; stepsize < blockDim.x; stepsize *= 2 )
        {
            __syncthreads();
            const auto value = ( value_index - stepsize >= 0 ) ? values[value_index - stepsize] : 0.0f;

            __syncthreads();
            values[value_index] += value;
        }
        integral_image[pixel_index] = values[value_index];
    }

    __global__ void compute_integral_triangles_a( const float2* integral_columns, float2* integral_triangles )
    {
        __shared__ float2 values[constants::maximum_resolution];

        const auto block        = static_cast<int>( blockIdx.x );
        const auto thread       = static_cast<int>( threadIdx.x );
        const auto coordinates  = make_int2(
            max( 0, block - ( static_cast<int>( blockDim.x ) - 1 ) ) + thread,
            max( 0, ( static_cast<int>( blockDim.x ) - 1 ) - block ) + thread );

        if( coordinates.x >= blockDim.x || coordinates.y >= blockDim.x )
        {
            return;
        }

        const auto pixel_index = coordinates.y * static_cast<int>( blockDim.x ) + coordinates.x;
        values[thread] = integral_columns[pixel_index];

        for( auto stepsize = 1; stepsize < blockDim.x; stepsize *= 2 )
        {
            __syncthreads();
            const auto x = ( coordinates.x - stepsize >= 0 && coordinates.y - stepsize >= 0 ) ? values[thread - stepsize].x : 0.0f;
            const auto y = ( coordinates.x + stepsize < blockDim.x && coordinates.y + stepsize < blockDim.x ) ? values[thread + stepsize].y : 0.0f;

            __syncthreads();
            values[thread].x += x;
            values[thread].y += y;
        }

        integral_triangles[pixel_index] = values[thread];
    }
    __global__ void compute_integral_triangles_b( const float2* integral_columns, float2* integral_triangles )
    {
        __shared__ float2 values[constants::maximum_resolution];

        const auto block        = static_cast<int>( blockIdx.x );
        const auto thread       = static_cast<int>( threadIdx.x );
        const auto coordinates  = make_int2(
            max( 0, block - ( static_cast<int>( blockDim.x ) - 1 ) ) + thread,
            min( ( static_cast<int>( blockDim.x ) - 1 ), block ) - thread );

        if( coordinates.x >= blockDim.x || coordinates.y < 0 )
        {
            return;
        }

        const auto pixel_index = coordinates.y * static_cast<int>( blockDim.x ) + coordinates.x;
        values[thread] = integral_columns[pixel_index];

        for( auto stepsize = 1; stepsize < blockDim.x; stepsize *= 2 )
        {
            __syncthreads();
            const auto x = ( coordinates.x + stepsize < blockDim.x && coordinates.y - stepsize >= 0 ) ? values[thread + stepsize].x : 0.0f;
            const auto y = ( coordinates.x - stepsize >= 0 && coordinates.y + stepsize < blockDim.x ) ? values[thread - stepsize].y : 0.0f;

            __syncthreads();
            values[thread].x += x;
            values[thread].y += y;
        }

        integral_triangles[pixel_index] = values[thread];
    }

    __global__ void compute_deformation_integral_images( const float* density_texture, const float2* integral_columns, const float* integral_image, const float2* integral_triangles_a, const float2* integral_triangles_b, int resolution, float2* deformation )
    {
        const auto x = static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x );
        const auto y = static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y );

        if( x > resolution || y > resolution )
        {
            return;
        }

        const auto position = make_float2(
            static_cast<float>( x ) / resolution,
            static_cast<float>( y ) / resolution
        );

        // Deformation from aligned integral images

        float2 q1, q2, q3, q4;

        if( position.y < position.x )
        {
            q1 = make_float2( 1.0f, 1.0f + position.y - position.x );
            q3 = make_float2( position.x - position.y, 0.0f );
        }
        else
        {
            q1 = make_float2( 1.0f - position.y + position.x, 1.0f );
            q3 = make_float2( 0.0f, position.y - position.x );
        }

        if( position.x + position.y < 1.0f )
        {
            q2 = make_float2( position.x + position.y, 0.0f );
            q4 = make_float2( 0.0f, position.x + position.y );
        }
        else
        {
            q2 = make_float2( 1.0f, position.x + position.y - 1.0f );
            q4 = make_float2( position.x + position.y - 1.0f, 1.0f );
        }

        const auto total    = utility::load( integral_image, make_int2( resolution - 1, resolution - 1 ), resolution );
        const auto left     = utility::load( integral_image, make_int2( x - 1, resolution - 1 ), resolution, 0.0f );
        const auto top      = utility::load( integral_image, make_int2( resolution - 1, y - 1 ), resolution, 0.0f );

        auto alpha          = utility::load( integral_image, make_int2( x - 1, y - 1 ), resolution, 0.0f );
        auto beta           = left - alpha;
        auto delta          = top - alpha;
        auto gamma          = total - alpha - beta - delta;
        auto weights        = make_float4( alpha / total, beta / total, gamma / total, delta / total );

        const auto density_deformation_aligned = make_float2(
            weights.x * q1.x + weights.y * q2.x + weights.z * q3.x + weights.w * q4.x,
            weights.x * q1.y + weights.y * q2.y + weights.z * q3.y + weights.w * q4.y
        );

        const auto ix       = 1.0f - position.x;
        const auto iy       = 1.0f - position.y;

        alpha               = position.x * position.y;
        beta                = position.x * iy;
        gamma               = ix * iy;
        delta               = ix * position.y;
        weights             = make_float4( alpha, beta, gamma, delta );

        const auto uniform_deformation_aligned = make_float2(
            weights.x * q1.x + weights.y * q2.x + weights.z * q3.x + weights.w * q4.x,
            weights.x * q1.y + weights.y * q2.y + weights.z * q3.y + weights.w * q4.y
        );

        // Deformation from tilted integral images

        q1 = make_float2( position.x, 1.0f );
        q2 = make_float2( 1.0f, position.y );
        q3 = make_float2( position.x, 0.0f );
        q4 = make_float2( 0.0f, position.y );

        auto topleft_top                = utility::load( integral_triangles_a, make_int2( x - 1, y - 1 ), resolution, make_float2( 0.0f, 0.0f ) ).x;
        const auto topleft_diagonal     = utility::load( integral_triangles_a, make_int2( x - 1, y - 2 ), resolution, make_float2( 0.0f, 0.0f ) ).x - topleft_top;
        topleft_top                     -= topleft_diagonal / 2.0f;

        auto topright_top               = utility::load( integral_triangles_b, make_int2( x, y - 1 ), resolution, make_float2( 0.0f, 0.0f ) ).x;
        const auto topright_diagonal    = utility::load( integral_triangles_b, make_int2( x, y - 2 ), resolution, make_float2( 0.0f, 0.0f ) ).x - topright_top;
        topright_top                    -= topright_diagonal / 2.0f;

        auto bottomleft_bottom          = utility::load( integral_triangles_b, make_int2( x - 1, y ), resolution, make_float2( 0.0f, 0.0f ) ).y;
        const auto bottomleft_diagonal  = utility::load( integral_triangles_b, make_int2( x - 1, y + 1 ), resolution, make_float2( 0.0f, 0.0f ) ).y - bottomleft_bottom;
        bottomleft_bottom               -= bottomleft_diagonal / 2.0f;

        auto bottomright_bottom         = utility::load( integral_triangles_a, make_int2( x, y ), resolution, make_float2( 0.0f, 0.0f ) ).y;
        const auto bottomright_diagonal = utility::load( integral_triangles_a, make_int2( x, y + 1 ), resolution, make_float2( 0.0f, 0.0f ) ).y - bottomright_bottom;
        bottomright_bottom              -= bottomright_diagonal / 2.0f;

        const auto topleft_bottom       = alpha - topleft_top;
        const auto bottomleft_top       = beta - bottomleft_bottom;

        alpha               = topleft_top + topright_top;
        beta                = topleft_bottom + bottomleft_top;
        gamma               = bottomleft_bottom + bottomright_bottom;
        delta               = total - alpha - beta - gamma;
        weights             = make_float4( alpha / total, beta / total, gamma / total, delta / total );

        const auto density_deformation_tilted = make_float2(
            weights.x * q1.x + weights.y * q2.x + weights.z * q3.x + weights.w * q4.x,
            weights.x * q1.y + weights.y * q2.y + weights.z * q3.y + weights.w * q4.y
        );

        if( position.x <= position.y )
        {
            if( position.x <= 1.0 - position.y )
            {
                beta    = position.x * position.x;
                alpha	= position.x * position.y - ( beta / 2.0 ) +  ( position.y * position.y / 2.0 );
                gamma	= position.x * iy - ( beta / 2.0 ) + ( iy * iy / 2.0 );
                delta	= 1.0 - alpha - beta - gamma;

            }
            else
            {
                gamma	= iy * iy;
                beta	= position.x * iy - ( gamma / 2.0 ) + ( position.x * position.x / 2.0 );
                delta	= ix * iy - ( gamma / 2.0 ) + ( ix * ix / 2.0 );
                alpha	= 1.0 - beta - gamma - delta;
            }
        }
        else
        {
            if( position.x <= 1.0 - position.y )
            {
                alpha	= position.y * position.y;
                beta	= position.x * position.y - ( alpha / 2.0 ) + ( position.x * position.x / 2.0 );
                delta	= ix * position.y - ( alpha / 2.0 ) + ( ix * ix / 2.0 );
                gamma	= 1.0 - alpha - beta - delta;

            }
            else
            {
                delta	= ix * ix;
                alpha	= ix * position.y - ( delta / 2.0 ) + ( position.y * position.y / 2.0 );
                gamma	= ix * iy - ( delta / 2.0 ) + ( iy * iy / 2.0 );
                beta	= 1.0 - alpha - gamma - delta;
            }
        }

        const auto uniform_deformation_tilted = make_float2(
            weights.x * q1.x + weights.y * q2.x + weights.z * q3.x + weights.w * q4.x,
            weights.x * q1.y + weights.y * q2.y + weights.z * q3.y + weights.w * q4.y
        );

        // Combined deformation

        const auto density_deformation = make_float2(
            ( density_deformation_aligned.x + density_deformation_tilted.x ) / 2.0f,
            ( density_deformation_aligned.y + density_deformation_tilted.y ) / 2.0f
        );
        const auto uniform_deformation = make_float2(
            ( uniform_deformation_aligned.x + uniform_deformation_tilted.x ) / 2.0f,
            ( uniform_deformation_aligned.y + uniform_deformation_tilted.y ) / 2.0f
        );

        deformation[y * ( resolution + 1 ) + x] = make_float2(
            density_deformation.x - uniform_deformation.x,
            density_deformation.y - uniform_deformation.y
        );
    }
    __global__ void compute_deformation_multiresolution( const float* density, int density_resolution, float2* deformation, int deformation_resolution )
    {
        __shared__ float shared_patch[17][17];

        const auto density_x_base = static_cast<int>( blockIdx.x * blockDim.x ) - 1;
        const auto density_y_base = static_cast<int>( blockIdx.y * blockDim.y ) - 1;

        for( int dx = threadIdx.x; dx < 17; dx += blockDim.x )
        {
            for( int dy = threadIdx.y; dy < 17; dy += blockDim.y )
            {
                const auto density_x = min( max( density_x_base + dx, 0 ), density_resolution - 1 );
                const auto density_y = min( max( density_y_base + dy, 0 ), density_resolution - 1 );
                shared_patch[dx][dy] = density[density_y * density_resolution + density_x];
            }
        }
        __syncthreads();

        const auto vertex_x             = static_cast<int>( blockIdx.x ) * blockDim.x + threadIdx.x;
        const auto vertex_y             = static_cast<int>( blockIdx.y ) * blockDim.y + threadIdx.y;
        if( vertex_x >= deformation_resolution || vertex_y >= deformation_resolution )
        {
            return;
        }

        const auto H00  = shared_patch[threadIdx.x][threadIdx.y];
        const auto H01  = shared_patch[threadIdx.x][threadIdx.y + 1];
        const auto H10  = shared_patch[threadIdx.x + 1][threadIdx.y];
        const auto H11  = shared_patch[threadIdx.x + 1][threadIdx.y + 1];
        const auto H    = H00 + H01 + H10 + H11;

        const auto denominator  = 4.0f * H;
        const auto dx           = ( H == 0.0f || vertex_x == 0 || vertex_x == density_resolution )? 0.0f : ( ( H00 + H01 - H10 - H11 ) / denominator / density_resolution );
        const auto dy           = ( H == 0.0f || vertex_y == 0 || vertex_y == density_resolution )? 0.0f : ( ( H00 + H10 - H01 - H11 ) / denominator / density_resolution );

        deformation[vertex_y * deformation_resolution + vertex_x] = make_float2( dx, dy );
    }

    __global__ void transform_points( const float2* deformation, int resolution, float2* points, int point_count )
    {
        for( int point_index = blockIdx.x * blockDim.x + threadIdx.x; point_index < point_count; point_index += blockDim.x * gridDim.x )
        {
            const auto point = points[point_index];

            const auto x_float = point.x * resolution;
            const auto y_float = point.y * resolution;

            const auto x = min( __float2int_rz( x_float ), resolution - 1 );
            const auto y = min( __float2int_rz( y_float ), resolution - 1 );

            const auto u = x_float - static_cast<float>( x );
            const auto v = y_float - static_cast<float>( y );

            const auto deformation00_index = y * ( resolution + 1 ) + x;
            const auto deformation10_index = deformation00_index + 1;
            const auto deformation01_index = deformation00_index + ( resolution + 1 );
            const auto deformation11_index = deformation01_index + 1;

            const auto deformation00 = __ldg( deformation + deformation00_index );
            const auto deformation10 = __ldg( deformation + deformation10_index );
            const auto deformation01 = __ldg( deformation + deformation01_index );
            const auto deformation11 = __ldg( deformation + deformation11_index );

            const auto weight00 = ( 1.0f - u ) * ( 1.0f - v );
            const auto weight10 = u * ( 1.0f - v );
            const auto weight01 = ( 1.0f - u ) * v;
            const auto weight11 = u * v;

            const auto dx = weight00 * deformation00.x + weight10 * deformation10.x + weight01 * deformation01.x + weight11 * deformation11.x;
            const auto dy = weight00 * deformation00.y + weight10 * deformation10.y + weight01 * deformation01.y + weight11 * deformation11.y;

            points[point_index] = make_float2( point.x + dx, point.y + dy );
        }
    }
}

// ----- Integral Images ----- //

void density_equalizing_transformation_integral_images(
    float* points,
    int point_count,
    int resolution,
    int kernel_radius,
    int iterations
)
{
    // Allocate device memory
    float2* points_device                   = nullptr;
    float* density_device                   = nullptr;
    float2* integral_columns_device         = nullptr;
    float* integral_image_device            = nullptr;
    float2* integral_triangles_a_device     = nullptr;
    float2* integral_triangles_b_device     = nullptr;
    float2* deformation_device              = nullptr;

    const auto points_bytes                 = point_count * sizeof( float2 );
    const auto density_bytes                = resolution * resolution * sizeof( float );
    const auto integral_columns_bytes       = resolution * resolution * sizeof( float2 );
    const auto integral_image_bytes         = resolution * resolution * sizeof( float );
    const auto integral_triangles_a_bytes   = resolution * resolution * sizeof( float2 );
    const auto integral_triangles_b_bytes   = resolution * resolution * sizeof( float2 );
    const auto deformation_resolution       = resolution + 1;
    const auto deformation_bytes            = deformation_resolution * deformation_resolution * sizeof( float2 );

    utility::assert_error( cudaMalloc( &points_device, points_bytes ), "Failed to allocate device memory for points" );
    utility::assert_error( cudaMalloc( &density_device, density_bytes ), "Failed to allocate device memory for density" );
    utility::assert_error( cudaMalloc( &integral_columns_device, integral_columns_bytes ), "Failed to allocate device memory for integral columns" );
    utility::assert_error( cudaMalloc( &integral_image_device, integral_image_bytes ), "Failed to allocate device memory for integral image" );
    utility::assert_error( cudaMalloc( &integral_triangles_a_device, integral_triangles_a_bytes ), "Failed to allocate device memory for integral triangles A" );
    utility::assert_error( cudaMalloc( &integral_triangles_b_device, integral_triangles_b_bytes ), "Failed to allocate device memory for integral triangles B" );
    utility::assert_error( cudaMalloc( &deformation_device, deformation_bytes ), "Failed to allocate device memory for deformation" );

    // Initialize kernel weights
    auto kernel_weights_device = utility::initialize_kernel_weights_device( kernel_radius );

    // Copy points to device
    utility::assert_error( cudaMemcpy( points_device, points, points_bytes, cudaMemcpyHostToDevice ), "Failed to copy points to device" );

    // Precompute background density, block sizes, grid sizes, and shared memory sizes
    const auto pixel_count				= resolution * resolution;
    const auto background_density		= static_cast<float>( point_count ) / pixel_count;

    const auto points_blocksize         = 512;
    const auto points_gridsize          = static_cast<unsigned int>( ( point_count + points_blocksize - 1 ) / points_blocksize );

    const auto shared_density_bytes     = ( resolution + 2 * kernel_radius ) * sizeof( float );

    const auto deformation_blocksize    = dim3 { 16, 16 };
    const auto deformation_gridsize     = dim3 {
        static_cast<unsigned int>( ( resolution + 1 + 16 - 1 ) / 16 ),
        static_cast<unsigned int>( ( resolution + 1 + 16 - 1 ) / 16 )
    };

    // Perform iterations
    for( auto iteration = 0; iteration < iterations; ++iteration )
    {
        // Reset density
        kernels::fill<<<108, 512>>>( density_device, pixel_count, background_density );

        // Compute density
        kernels::compute_density_global_atomic<<<points_gridsize, points_blocksize>>>( points_device, point_count, density_device, resolution );

        // Smooth density
        if( kernel_radius > 0 )
        {
            kernels::smooth_density_horizontal<<<resolution, resolution, shared_density_bytes>>>( density_device, resolution, kernel_weights_device, kernel_radius );
            kernels::smooth_density_vertical<<<resolution, resolution, shared_density_bytes>>>( density_device, resolution, kernel_weights_device, kernel_radius );
        }

        // Compute integral columns
        kernels::compute_integral_columns<<<resolution, resolution>>>( density_device, integral_columns_device );

        // Compute integral image
        kernels::compute_integral_image<<<resolution, resolution>>>( integral_columns_device, integral_image_device );

        // Compute integral triangles A
        kernels::compute_integral_triangles_a<<<2 * resolution - 1, resolution>>>( integral_columns_device, integral_triangles_a_device );

        // Compute integral triangles B
        kernels::compute_integral_triangles_b<<<2 * resolution - 1, resolution>>>( integral_columns_device, integral_triangles_b_device );

        // Compute deformation
        kernels::compute_deformation_integral_images<<<deformation_gridsize, deformation_blocksize>>>(
            density_device, integral_columns_device, integral_image_device, integral_triangles_a_device, integral_triangles_b_device, resolution, deformation_device
        );

        // Transform points
        kernels::transform_points<<<points_gridsize, points_blocksize>>>( deformation_device, resolution, points_device, point_count );
    }

    // Copy points to host
    utility::assert_error( cudaMemcpy( points, points_device, points_bytes, cudaMemcpyDeviceToHost ), "Failed to copy points from device" );

    // Free device memory
    utility::assert_error( cudaFree( deformation_device ), "Failed to free device memory for deformation" );
    utility::assert_error( cudaFree( integral_triangles_b_device ), "Failed to free device memory for integral triangles B" );
    utility::assert_error( cudaFree( integral_triangles_a_device ), "Failed to free device memory for integral triangles A" );
    utility::assert_error( cudaFree( integral_image_device ), "Failed to free device memory for integral image" );
    utility::assert_error( cudaFree( integral_columns_device ), "Failed to free device memory for integral columns" );
    utility::assert_error( cudaFree( density_device ), "Failed to free device memory for density" );
    utility::assert_error( cudaFree( kernel_weights_device ), "Failed to free device memory for kernel weights" );
    utility::assert_error( cudaFree( points_device ), "Failed to free device memory for points" );
}

// ----- Sector-based ----- //

void density_equalizing_transformation_sector_based(
    float* points,
    int point_count,
    int sector_count,
    int iterations
)
{
    // Precompute sector vectors
    struct Sector
    {
        float2 vector_begin     = make_float2( 0.0f, 0.0f );
        float2 vector_anchor    = make_float2( 0.0f, 0.0f );
        float2 vector_end       = make_float2( 0.0f, 0.0f );
    };

    auto sectors = std::vector<Sector>( sector_count );

    const auto sector_radian_step = constants::two_pi / sector_count;
    for( auto sector_index = 0; sector_index < sector_count; ++sector_index )
    {
        auto& sector = sectors[sector_index];

        const auto sector_radian_begin  = ( sector_index + 0.0f ) * sector_radian_step;
        const auto sector_radian_anchor = ( sector_index + 0.5f ) * sector_radian_step;
        const auto sector_radian_end    = ( sector_index + 1.0f ) * sector_radian_step;

        sector.vector_begin     = make_float2( std::cosf( sector_radian_begin ), std::sinf( sector_radian_begin ) );
        sector.vector_anchor    = make_float2( -std::cosf( sector_radian_anchor ), -std::sinf( sector_radian_anchor ) );
        sector.vector_end       = make_float2( std::cosf( sector_radian_end ), std::sinf( sector_radian_end ) );
    }

    // Utility functions for intersection and area computation
    enum class Side
    {
        eNone, eLeft, eRight, eBottom, eTop
    };

    struct Intersection
    {
        float2 point    = make_float2( 0.0f, 0.0f );
        Side side       = Side::eNone;
    };

    const auto compute_intersection = [] ( float2 point, float2 direction )
    {
        auto intersection = Intersection {};
        auto minimum_t = std::numeric_limits<float>::max();

        if( direction.x > 0.0f )
        {
            const auto t = ( 1.0f - point.x ) / direction.x;
            if( t < minimum_t )
            {
                minimum_t = t;
                intersection.side = Side::eRight;
            }
        }
        else if( direction.x < 0.0f )
        {
            const auto t = -point.x / direction.x;
            if( t < minimum_t )
            {
                minimum_t = t;
                intersection.side = Side::eLeft;
            }
        }

        if( direction.y > 0.0f )
        {
            const auto t = ( 1.0f - point.y ) / direction.y;
            if( t < minimum_t )
            {
                minimum_t = t;
                intersection.side = Side::eTop;
            }
        }
        else if( direction.y < 0.0f )
        {
            const auto t = -point.y / direction.y;
            if( t < minimum_t )
            {
                minimum_t = t;
                intersection.side = Side::eBottom;
            }
        }

        intersection.point = make_float2( point.x + minimum_t * direction.x, point.y + minimum_t * direction.y );

        if( intersection.side == Side::eLeft )          intersection.point.x = 0.0f;
        else if( intersection.side == Side::eRight )    intersection.point.x = 1.0f;
        else if( intersection.side == Side::eBottom )   intersection.point.y = 0.0f;
        else if( intersection.side == Side::eTop )      intersection.point.y = 1.0f;

        return intersection;
    };
    const auto compute_triangle_area = [] ( float2 a, float2 b, float2 c )
    {
        return std::abs( a.x * ( b.y - c.y ) + b.x * ( c.y - a.y ) + c.x * ( a.y - b.y ) ) / 2.0f;
    };

    // Utility function to compute deformations in parallel
    auto deformations = std::vector<float2>( point_count, make_float2( 0.0f, 0.0f ) );
    const auto compute_deformation = [&] ( int point_index, int thread_count )
    {
        for( ; point_index < point_count; point_index += thread_count )
        {
            const auto point = make_float2( points[2 * point_index], points[2 * point_index + 1] );

            // Count points in sectors
            auto point_counts = std::vector<int>( sector_count, 0 );
            for( auto point_index_other = 0; point_index_other < point_count; ++point_index_other )
            {
                const auto point_other  = make_float2( points[2 * point_index_other], points[2 * point_index_other + 1] );
                const auto direction    = make_float2( point.x - point_other.x, point.y - point_other.y );

                if( direction.x == 0.0f && direction.y == 0.0f )
                {
                    continue;
                }

                const auto radian   = std::atan2f( direction.y, direction.x ) + constants::pi;
                auto sector_index   = static_cast<int>( radian / constants::two_pi * sector_count );
                if( sector_index >= sector_count )
                {
                    sector_index = 0;
                }

                point_counts[sector_index]++;
            }

            auto point_counts_total = 0;
            for( const auto point_count : point_counts )
            {
                point_counts_total += point_count;
            }

            if( point_counts_total == 0 )
            {
                return;
            }

            // Compute deformation
            auto& deformation = deformations[point_index] = make_float2( 0.0f, 0.0f );
            for( auto sector_index = 0; sector_index < sector_count; ++sector_index )
            {
                // Compute sector area
                const auto& sector              = sectors[sector_index];
                const auto intersection_begin   = compute_intersection( point, sector.vector_begin );
                const auto intersection_end     = compute_intersection( point, sector.vector_end );

                auto sector_area = compute_triangle_area( point, intersection_begin.point, intersection_end.point );
                if( intersection_begin.side != intersection_end.side )
                {
                    auto corner = make_float2( 0.0f, 0.0f );
                    if( intersection_begin.side == Side::eLeft ) corner = make_float2( 0.0f, 0.0f );
                    else if( intersection_begin.side == Side::eRight ) corner = make_float2( 1.0f, 1.0f );
                    else if( intersection_begin.side == Side::eBottom ) corner = make_float2( 1.0f, 0.0f );
                    else if( intersection_begin.side == Side::eTop ) corner = make_float2( 0.0f, 1.0f );
                    sector_area += compute_triangle_area( intersection_begin.point, intersection_end.point, corner );
                }


                // Compute weights
                const auto weight_density   = static_cast<float>( point_counts[sector_index] ) / point_counts_total;
                const auto weight_uniform   = sector_area;
                const auto weight_combined  = weight_density - weight_uniform;

                // Update deformation
                const auto anchor           = compute_intersection( point, sector.vector_anchor );
                deformation.x               += weight_combined * anchor.point.x;
                deformation.y               += weight_combined * anchor.point.y;
            }
        }
    };

    // Prepare threads
    const auto thread_count = static_cast<int>( std::thread::hardware_concurrency() );
    auto threads = std::vector<std::thread>( thread_count );

    // Perform iterations
    for( auto iteration = 0; iteration < iterations; ++iteration )
    {
        // Compute deformations
        for( auto thread_index = 0; thread_index < thread_count; ++thread_index )
        {
            threads[thread_index] = std::thread { compute_deformation, thread_index, thread_count };
        }
        for( auto& thread : threads )
        {
            thread.join();
        }

        // Transform points
        for( auto point_index = 0; point_index < point_count; ++point_index )
        {
            const auto deformation = deformations[point_index];
            points[2 * point_index]         += deformation.x;
            points[2 * point_index + 1]     += deformation.y;
        }
    }
}

// ----- Multiresolution ----- //

void density_equalizing_transformation_multiresolution(
    float* points,
    int point_count,
    int maximum_resolution,
    int kernel_radius,
    int cycle_count
)
{
    // Allocate device memory
    float2* points_device                       = nullptr;
    float* density_device                       = nullptr;
    float2* deformation_device                  = nullptr;

    const auto points_bytes                     = point_count * sizeof( float2 );
    const auto maximum_density_bytes            = maximum_resolution * maximum_resolution * sizeof( float );
    const auto maximum_deformation_resolution   = maximum_resolution + 1;
    const auto maximum_deformation_bytes        = maximum_deformation_resolution * maximum_deformation_resolution * sizeof( float2 );

    utility::assert_error( cudaMalloc( &points_device, points_bytes ), "Failed to allocate device memory for points" );
    utility::assert_error( cudaMalloc( &density_device, maximum_density_bytes ), "Failed to allocate device memory for density" );
    utility::assert_error( cudaMalloc( &deformation_device, maximum_deformation_bytes ), "Failed to allocate device memory for deformation" );

    // Initialize kernel weights
    auto kernel_weights_device = utility::initialize_kernel_weights_device( kernel_radius );

    // Copy points to device
    utility::assert_error( cudaMemcpy( points_device, points, points_bytes, cudaMemcpyHostToDevice ), "Failed to copy points to device" );

    // Precompute block sizes, grid sizes, and shared memory sizes
    const auto points_blocksize         = 512;
    const auto points_gridsize          = static_cast<unsigned int>( ( point_count + points_blocksize - 1 ) / points_blocksize );

    // Perform cycles
    for( auto cycle_index = 0; cycle_index < cycle_count; ++cycle_index )
    {
        for( auto resolution = maximum_resolution; resolution >= 2; resolution /= 2 )
        {
            // Reset density
            const auto density_bytes = resolution * resolution * sizeof( float );
            cudaMemset( density_device, 0, density_bytes );

            // Compute density
            if( ( resolution < 64 && point_count >= 10'000 ) || ( resolution == 64 && point_count >= 100'000 ) )
            {
                kernels::compute_density_shared_atomic<<<points_gridsize, points_blocksize, density_bytes>>>(
                    points_device, point_count, density_device, resolution
                );
            }
            else
            {
                kernels::compute_density_global_atomic<<<points_gridsize, points_blocksize>>>(
                    points_device, point_count, density_device, resolution
                );
            }

            // Smooth density
            if( kernel_radius > 0 )
            {
                const auto shared_density_bytes = ( resolution + 2 * kernel_radius ) * sizeof( float );
                kernels::smooth_density_horizontal<<<resolution, resolution, shared_density_bytes>>>( density_device, resolution, kernel_weights_device, kernel_radius );
                kernels::smooth_density_vertical<<<resolution, resolution, shared_density_bytes>>>( density_device, resolution, kernel_weights_device, kernel_radius );
            }

            // Compute deformation
            const auto deformation_blocksize    = dim3 { 16, 16 };
            const auto deformation_gridsize     = dim3 {
                static_cast<unsigned int>( ( resolution + 1 + 16 - 1 ) / 16 ),
                static_cast<unsigned int>( ( resolution + 1 + 16 - 1 ) / 16 )
            };
            kernels::compute_deformation_multiresolution<<<deformation_gridsize, deformation_blocksize>>>( density_device, resolution, deformation_device, resolution + 1 );

            // Transform points
            kernels::transform_points<<<points_gridsize, points_blocksize>>>( deformation_device, resolution, points_device, point_count );
        }
    }

    // Copy points to host
    utility::assert_error( cudaMemcpy( points, points_device, points_bytes, cudaMemcpyDeviceToHost ), "Failed to copy points from device" );

    // Free device memory
    utility::assert_error( cudaFree( deformation_device ), "Failed to free device memory for deformation" );
    utility::assert_error( cudaFree( density_device ), "Failed to free device memory for density" );
    utility::assert_error( cudaFree( kernel_weights_device ), "Failed to free device memory for kernel weights" );
    utility::assert_error( cudaFree( points_device ), "Failed to free device memory for points" );
}