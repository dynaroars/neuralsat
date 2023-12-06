
# Typical use (collects GPU timeline, Cuda and OS calls on the CPU timeline, but no CPU stack traces)
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o nsight_report -f true -x true python script.py args...

# Adds CPU backtraces that will show when you mouse over a long call or small orange tick (sample) on the CPU timeline:
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o nsight_report -f true --cudabacktrace=true --cudabacktrace-threshold=10000 --osrt-threshold=10000 -x true python script.py args...

# Focused profiling, profiles only a target region
# (your app must call torch.cuda.cudart().cudaProfilerStart()/Stop() at the start/end of the target region)
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o nsight_report -f true --capture-range=cudaProfilerApi --stop-on-range-end=true --cudabacktrace=true --cudabacktrace-threshold=10000 --osrt-threshold=10000 -x true python script.py args...

# if appname creates child processes, nsys WILL profile those as well.  They will show up as separate processes with
# separate timelines when you open the profile in nsight-sys

# Breakdown of options:
nsys profile
-w true # Don't suppress app's console output.
-t cuda,nvtx,osrt,cudnn,cublas # Instrument, and show timeline bubbles for, cuda api calls, nvtx ranges,
                               # os runtime functions, cudnn library calls, and cublas library calls.
                               # These options do not require -s cpu nor do they silently enable -s cpu.
-s cpu # Sample the cpu stack periodically.  Stack samples show up as little tickmarks on the cpu timeline.
       # Last time i checked they were orange, but still easy to miss.
       # Mouse over them to show the backtrace at that point.
       # -s cpu can increase cpu overhead substantially (I've seen 2X or more) so be aware of that distortion.
       # -s none disables cpu sampling.  Without cpu sampling, the profiling overhead is reduced.
       # Use -s none if you want the timeline to better represent a production job (api calls and kernels will
       # still appear on the profile, but profiling them doesn't distort the timeline nearly as much).
-o nsight_report # output file
-f true # overwrite existing output file
--capture-range=cudaProfilerApi # Only start profiling when the app calls cudaProfilerStart...
--stop-on-range-end=true # ...and end profiling when the app calls cudaProfilerStop.
--cudabacktrace=true # Collect a cpu stack sample for cuda api calls whose runtime exceeds some threshold.
                     # When you mouse over a long-running api call on the timeline, a backtrace will
                     # appear, and you can identify which of your functions invoked it.
                     # I really like this feature.
                     # Requires -s cpu.
--cudabacktrace-threshold=10000 # Threshold (in nanosec) that determines how long a cuda api call
                                # must run to trigger a backtrace.  10 microsec is a reasonable value
                                # (most kernel launches should take less than 10 microsec) but you
                                # should retune if you see a particular api call you'd like to investigate.
                                # Requires --cudabacktrace=true and -s cpu.
--osrt-threshold=10000 # Threshold (in nanosec) that determines how long an os runtime call (eg sleep)
                       # must run to trigger a backtrace.
                       # Backtrace collection for os runtime calls that exceed this threshold should
                       # occur by default if -s cpu is enabled.
-x true # Quit the profiler when the app exits.
python script.py args...