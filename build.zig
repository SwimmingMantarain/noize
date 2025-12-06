const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const use_cl = b.option(bool, "use_cl", "Use OpenCL") orelse false;

    const options = b.addOptions();
    options.addOption(bool, "use_cl", use_cl);

    const lib = b.addModule("noize", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    lib.addOptions("config", options);
}
