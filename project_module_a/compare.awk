BEGIN {
    max_err = 0
    sum_sq = 0
    count = 0
}
NR == FNR {
    serial[NR] = $2
    next
}
{
    diff = serial[FNR] - $2
    abs_diff = (diff < 0) ? -diff : diff
    if (abs_diff > max_err) max_err = abs_diff
    sum_sq += diff * diff
    count++
}
END {
    l2_err = sqrt(sum_sq)
    printf "Maximum absolute error: %.6e\n", max_err
    printf "L2 norm of error:       %.6e\n", l2_err
    printf "Number of points:       %d\n", count
    if (max_err < 1e-4) {
        print "\n✓ Solutions MATCH (error < 1e-4)"
    } else if (max_err < 1e-2) {
        print "\n⚠ Solutions are CLOSE but with moderate error"
    } else {
        print "\n✗ Solutions DIFFER significantly"
    }
}
