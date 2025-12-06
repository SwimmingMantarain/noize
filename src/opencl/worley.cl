#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef struct {
    uint biome_id;
	uint _pad;
    double percent;
} BiomeBlendCL;

ulong hash_xy_cl(ulong seed, int x, int y) {
    ulong packed_coords = ((ulong)((uint)x)) | (((ulong)((uint)y)) << 32);
    
    ulong key = seed ^ packed_coords;

    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    key = key + (key << 31);

    return key;
}

double warp_cl(double coord, ulong seed, double strength) {
    // Simple pseudo-random float based on coordinate
    ulong h = hash_xy_cl(seed, (int)(coord * 1000.0), 0);
    double r = (double)(h & 0xFFFF) / 65535.0; // 0.0 to 1.0
    return coord * r * strength;
}

void cell_point_cl(ulong seed, int x, int y, double* out_x, double* out_y) {
    ulong hashx = hash_xy_cl(seed + 777UL, x, y);
    ulong hashy = hash_xy_cl(seed + 7777UL, x, y);

    double offset_x = (double)(hashx & 0xFFFF) / 65535.0;
    double offset_y = (double)(hashy & 0xFFFF) / 65535.0;

    *out_x = (double)x + offset_x;
    *out_y = (double)y + offset_y;
}


__kernel void worley_blend(
    __global BiomeBlendCL* output_blends,
    const ulong seed,
    const double sharpness,
    const double zoom,
    const double world_x_offset,
    const double world_z_offset,
    const uint num_biomes,
    const double warp_strength
) {
    uint gx = get_global_id(0); 
    uint gz = get_global_id(1);

    const double world_x_value = world_x_offset + (double)gx;
    const double world_z_value = world_z_offset + (double)gz;

    double x_unwarped = world_x_value / zoom;
    double z_unwarped = world_z_value / zoom;

    const double x = x_unwarped; // warp_cl(x_unwarped, seed, warp_strength);
    const double z = z_unwarped; // warp_cl(z_unwarped, seed, warp_strength);

    const int cell_x = (int)floor(x);
    const int cell_z = (int)floor(z);

    const int NEIGHBOR_OFFSETS[9][2] = {
        {-1,  1}, {0,  1}, {1,  1},
        {-1,  0}, {0,  0}, {1,  0},
        {-1, -1}, {0, -1}, {1, -1},
    };

    BiomeBlendCL local_blends[9];
    double sum_weight = 0.0;
    
    for (int i = 0; i < 9; ++i) {
        const int cx = cell_x + NEIGHBOR_OFFSETS[i][0];
        const int cz = cell_z + NEIGHBOR_OFFSETS[i][1];
        
        double point_x, point_z;
        cell_point_cl(seed, cx, cz, &point_x, &point_z);

        ulong random_val = hash_xy_cl(seed, cx, cz);
        uint biome_int = (uint)(random_val) % num_biomes;

        double dx = x - point_x;
        double dz = z - point_z;
        double dist = dx * dx + dz * dz;

        double w = 0.0;
        if (dist < 1e-9) {
            w = 100.0;
        } else {
            w = 1.0 / pow(dist, sharpness);
        }

        sum_weight += w;
        
        local_blends[i].biome_id = biome_int;
		local_blends[i]._pad = 0;
        local_blends[i].percent = w;
    }
    
    size_t output_start_index = (gz * 32 + gx) * 9; 

    for (int i = 0; i < 9; ++i) {
        output_blends[output_start_index + i].biome_id = local_blends[i].biome_id;
		output_blends[output_start_index + i]._pad = 0;

        if (sum_weight < 1e-9) sum_weight = 1.0;
        output_blends[output_start_index + i].percent = local_blends[i].percent / sum_weight;
    }
}
