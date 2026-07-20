const build = @import("build.zig");
const LazyPath = @import("std").Build.LazyPath;

// Copy this file to `build_config.zig` and edit the paths for your machine.
//
// `build.zig` reads `config` and wires each field into the root module:
//   - `system_libraries` -> `linkSystemLibrary` (e.g. "gsl", "blas")
//   - `frameworks`       -> `linkFramework`      (macOS, e.g. "Accelerate")
//   - `include_paths`    -> `addIncludePath`     (where C headers live)
//   - `library_paths`    -> `addLibraryPath`     (where the .dylib/.so live)
//
// The example below links the GNU Scientific Library (GSL) alongside the
// project's existing BLAS/LAPACK/TBLIS setup. GSL is used by `src/gsl.zig`
// (the `rand` and `stats` namespaces).

pub const config = build.Config{
    // macOS provides BLAS/LAPACK through the Accelerate framework.
    .frameworks = &[_][]const u8{"Accelerate"},

    // `gsl` pulls in libgsl. GSL also needs a CBLAS; here we satisfy that with
    // the system `blas` (Accelerate), so `gslcblas` is not required. If you are
    // not on macOS, add "gslcblas" and drop the Accelerate framework/`blas`.
    .system_libraries = &[_][]const u8{ "blas", "tblis", "lapack", "gsl" },

    // Header search paths. On Apple Silicon, Homebrew installs GSL headers to
    // `/opt/homebrew/include` (so `#include <gsl/gsl_rng.h>` resolves). On Intel
    // macOS Homebrew this is `/usr/local/include`.
    .include_paths = &[_]LazyPath{
        .{ .cwd_relative = "/path/to/tblis/build/include/" },
        .{ .cwd_relative = "/opt/homebrew/include/" },
    },

    // Library search paths. Homebrew installs libgsl to `/opt/homebrew/lib`
    // (Apple Silicon) or `/usr/local/lib` (Intel).
    //
    // Tip: run `gsl-config --cflags --libs` to print the exact include/lib
    // flags for your GSL installation.
    .library_paths = &[_]LazyPath{
        .{ .cwd_relative = "/path/to/tblis/build/lib/" },
        .{ .cwd_relative = "/opt/homebrew/lib/" },
    },
};
