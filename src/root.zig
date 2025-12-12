const std = @import("std");
const config = @import("config");
const cl = if (config.use_cl) @import("cl").cl else null;

pub const Noise = enum {
    Worley,
    Perlin,
};

const NEIGHBOR_OFFSETS: [9][2]i32 = .{
    [2]i32{ -1, 1 },  [2]i32{ 0, 1 },  [2]i32{ 1, 1 },
    [2]i32{ -1, 0 },  [2]i32{ 0, 0 },  [2]i32{ 1, 0 },
    [2]i32{ -1, -1 }, [2]i32{ 0, -1 }, [2]i32{ 1, -1 },
};

pub fn BiomeBlend(comptime E: type) type {
    return struct {
        biome: E,
        percent: f64,
    };
}

pub const BiomeBlendCL = extern struct {
    biome_id: u32,
    _pad: u32 = 0,
    percent: f64,
};

pub const Gen = struct {
    alloc: std.mem.Allocator,
    type: Noise,
    seed: u64,
    sharpness: f64,
    warp_strength: f64,
    zoom: f64,
    oc: ?*anyopaque = null,

    pub fn WorleyBlend32x32(self: *const @This(), gen: *Gen, comptime E: type, world_x_offset: i32, world_z_offset: i32) ![]BiomeBlendCL {
        comptime if (@typeInfo(E) != .@"enum") @compileError("Expected enum type");

        const num_biomes = @typeInfo(E).@"enum".fields.len;
        const output_size_coords = 32 * 32;
        const blends_per_coord = 9;
        const total_blend_count = output_size_coords * blends_per_coord;
        const output_buffer_size = total_blend_count * @sizeOf(BiomeBlendCL);

        var err: cl.cl_int = undefined;

        const output_mem = cl.clCreateBuffer(self.cl_context, cl.CL_MEM_WRITE_ONLY, output_buffer_size, null, &err);
        if (err != cl.CL_SUCCESS) return error.OpenClBufferFailed;
        defer _ = cl.clReleaseMemObject(output_mem);

        const zero = BiomeBlendCL{ .biome_id = 0, .percent = 0.0 };
        err = cl.clEnqueueFillBuffer(self.cl_queue, output_mem, &zero, @sizeOf(BiomeBlendCL), 0, output_buffer_size, 0, null, null);
        if (err != cl.CL_SUCCESS) return error.OpenCLWriteBuffFailed;

        err = cl.clSetKernelArg(self.worley_kernel, 0, @sizeOf(cl.cl_mem), @ptrCast(&output_mem));
        if (err != cl.CL_SUCCESS) return error.OpenCLKernelArg0Failed;

        err = cl.clSetKernelArg(self.worley_kernel, 1, @sizeOf(u64), &gen.seed);
        if (err != cl.CL_SUCCESS) return error.OpenCLKernelArg1Failed;

        err = cl.clSetKernelArg(self.worley_kernel, 2, @sizeOf(f64), &gen.sharpness);
        if (err != cl.CL_SUCCESS) return error.OpenCLKernelArg2Failed;

        err = cl.clSetKernelArg(self.worley_kernel, 3, @sizeOf(f64), &gen.zoom);
        if (err != cl.CL_SUCCESS) return error.OpenCLKernelArg3Failed;

        err = cl.clSetKernelArg(self.worley_kernel, 4, @sizeOf(f64), &@as(f64, @floatFromInt(world_x_offset)));
        if (err != cl.CL_SUCCESS) return error.OpenCLKernelArg4Failed;

        err = cl.clSetKernelArg(self.worley_kernel, 5, @sizeOf(f64), &@as(f64, @floatFromInt(world_z_offset)));
        if (err != cl.CL_SUCCESS) return error.OpenCLKernelArg5Failed;

        const num_biomes_u32: u32 = @intCast(num_biomes);
        err = cl.clSetKernelArg(self.worley_kernel, 6, @sizeOf(u32), &num_biomes_u32);
        if (err != cl.CL_SUCCESS) return error.OpenCLKernelArg6Failed;

        err = cl.clSetKernelArg(self.worley_kernel, 7, @sizeOf(f64), &gen.warp_strength);
        if (err != cl.CL_SUCCESS) return error.OpenCLKernelArg7Failed;

        const global_work_size = [2]usize{ 32, 32 }; // 32 * 32 = 1024 work items
        err = cl.clEnqueueNDRangeKernel(
            self.cl_queue,
            self.worley_kernel,
            2,
            null,
            &global_work_size,
            null,
            0,
            null,
            null,
        );
        if (err != cl.CL_SUCCESS) return error.OpenClKernelExecutionFailed;

        const host_buffer = try gen.alloc.alloc(BiomeBlendCL, total_blend_count);
        errdefer gen.alloc.free(host_buffer);

        err = cl.clEnqueueReadBuffer(
            self.cl_queue,
            output_mem,
            cl.CL_TRUE,
            0,
            output_buffer_size,
            host_buffer.ptr,
            0,
            null,
            null,
        );
        if (err != cl.CL_SUCCESS) {
            std.log.err("Failed to read OpenCL buffer: {}", .{err});
            return error.OpenClReadBufferFailed;
        }

        return host_buffer;
    }

    pub fn init(self: *Gen) !void {
        if (!config.use_cl) return;

        var err: cl.cl_int = undefined;
        const worley_program_src = @embedFile("./opencl/worley.cl");
        const worley_program = cl.clCreateProgramWithSource(
            self.opencl.?.cl_context,
            1,
            @ptrCast(@constCast(&worley_program_src)),
            null,
            &err,
        );

        if (err != cl.CL_SUCCESS) {
            std.debug.print("noize: Failed to create program: {}\n", .{err});
            return error.OpenClProgramFailed;
        }

        err = cl.clBuildProgram(worley_program, 1, &self.opencl.?.cl_devices, null, null, null);
        if (err != cl.CL_SUCCESS) {
            var log_size: usize = 0;
            _ = cl.clGetProgramBuildInfo(worley_program, self.opencl.?.cl_devices, cl.CL_PROGRAM_BUILD_LOG, 0, null, &log_size);
            const buf = try self.alloc.alloc(u8, log_size);
            defer self.alloc.free(buf);
            _ = cl.clGetProgramBuildInfo(worley_program, self.opencl.?.cl_devices, cl.CL_PROGRAM_BUILD_LOG, log_size, buf.ptr, null);
            std.debug.print("noize: Failed to build program: {s}", .{buf.ptr[0..buf.len]});
            return;
        }

        self.opencl.?.worley_program = worley_program;

        const worley_kernel = cl.clCreateKernel(worley_program, "worley_blend", &err);
        if (err != cl.CL_SUCCESS) {
            std.debug.print("noize: Failed to create worley kernel: {}", .{err});
            return;
        }

        self.opencl.?.worley_kernel = worley_kernel;
    }

    pub fn deinit(self: *Gen) void {
        if (!config.use_cl) return;

        _ = cl.clReleaseProgram(self.opencl.?.worley_program);
    }

    fn noise(self: *Gen, comptime T: type, x_value: f64, y_value: f64) T {
        switch (self.type) {
            .Worley => {
                const x_unwarped = x_value / self.zoom;
                const y_unwarped = y_value / self.zoom;

                const x = warp(x_unwarped);
                const y = warp(y_unwarped);

                const cell_x: i32 = @intFromFloat(@floor(x));
                const cell_y: i32 = @intFromFloat(@floor(y));

                var closest = std.math.floatMax(f64);
                var closest_biome_int: usize = 0;

                for (NEIGHBOR_OFFSETS[0..8]) |offset| {
                    const cx = cell_x + offset[0];
                    const cy = cell_y + offset[1];
                    const points = cell_point(self.seed, cx, cy);

                    if (@typeInfo(T) == .@"enum") {
                        const random_val = hash_xy(self.seed, cx, cy);
                        const biomes = @typeInfo(T).@"enum".fields;

                        const dist = std.math.pow(f64, x - points[0], 2) + std.math.pow(f64, y - points[1], 2);
                        if (dist < closest) {
                            closest_biome_int = @as(usize, @intCast(random_val)) % biomes.len;
                            closest = dist;
                        }

                        continue;
                    }

                    const dist = std.math.pow(f64, x - points[0], 2) + std.math.pow(f64, y - points[1], 2);
                    if (dist < closest) closest = dist;
                }

                return if (@typeInfo(T) == .@"enum") @enumFromInt(closest_biome_int) else closest;
            },
            else => unreachable,
        }
    }

    pub fn noisef64(self: *Gen, x: f64, y: f64) f64 {
        return self.noise(f64, x, y);
    }

    pub fn noiseT(self: *Gen, comptime E: type, x: f64, y: f64) E {
        comptime if (@typeInfo(E) != .@"enum") @compileError("Expected enum type");

        return self.noise(E, x, y);
    }

    pub fn noiseTBlend(self: *Gen, comptime E: type, x_value: f64, y_value: f64) ![]BiomeBlend(E) {
        comptime if (@typeInfo(E) != .@"enum") @compileError("Expected enum type");

        const x = x_value / self.zoom;
        const y = y_value / self.zoom;

        const cell_x: i32 = @intFromFloat(@floor(x));
        const cell_y: i32 = @intFromFloat(@floor(y));

        var blends = std.ArrayList(BiomeBlend(E)).initCapacity(self.alloc, 64) catch {
            std.log.err("Failed to alloc memory for biome blending", .{});
            return error.FailedBlender;
        };
        errdefer blends.deinit(self.alloc);

        const biomes = @typeInfo(E).@"enum".fields;
        var sum_weight: f64 = 0.0;

        for (NEIGHBOR_OFFSETS[0..8]) |offset| {
            const cx = cell_x + offset[0];
            const cy = cell_y + offset[1];
            const points = cell_point(self.seed, cx, cy);

            const random_val = hash_xy(self.seed, cx, cy);
            const biome_int = @as(usize, @intCast(random_val)) % biomes.len;
            const biome: E = @enumFromInt(biome_int);

            const dist = std.math.pow(f64, x - points[0], 2) + std.math.pow(f64, y - points[1], 2);

            const w = if (dist < 1e-9) 100.0 else 1.0 / std.math.pow(f64, dist, self.sharpness);
            sum_weight += w;

            try blends.append(self.alloc, .{
                .biome = biome,
                .percent = w,
            });
        }

        for (blends.items) |*blend| blend.percent /= sum_weight;

        return try blends.toOwnedSlice(self.alloc);
    }
};

fn cell_point(seed: u64, x: i32, y: i32) [2]f64 {
    const hashx = hash_xy(seed +% 777, x, y); // +% twos-complement wrapping on integer overflow
    const hashy = hash_xy(seed +% 7777, x, y);

    const offset_x = @as(f64, @floatFromInt(hashx & 65535)) / 65535.0;
    const offset_y = @as(f64, @floatFromInt(hashy & 65535)) / 65535.0;
    return .{ @as(f64, @floatFromInt(x)) + offset_x, @as(f64, @floatFromInt(y)) + offset_y };
}

fn hash_xy(seed: u64, x: i32, y: i32) u64 {
    const x_u64: u64 = @intCast(@as(u32, @bitCast(x)));
    const y_u64: u64 = @intCast(@as(u32, @bitCast(y)));

    const packed_coords = x_u64 | (y_u64 << 32);

    var key = seed ^ packed_coords;

    // Wang Hash
    key = (~key) +% (key << 21);
    key = key ^ (key >> 24);
    key = (key +% (key << 3)) +% (key << 8);
    key = key ^ (key >> 14);
    key = (key +% (key << 2)) +% (key << 4);
    key = key ^ (key >> 28);
    key = key +% (key << 31);

    return key;
}

fn warp(gen: *Gen, coord: f64) f64 {
    const rand = std.Random.DefaultPrng.init(gen.seed);
    return coord * rand.random().float(f64) * gen.warp_strength;
}
