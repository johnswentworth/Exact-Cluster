{
  "targets": [
    {
      "target_name": "cluster",
      "sources": [ "exact-cluster/cluster.cc" ],
      "cflags_cc+": [ "-std=c++0x", "-fpermissive", "-static", "-static-libgcc", "-static-libstdc++"],
      "include_dirs" : ["<!(node -e \"require('nan')\")"],
    }
  ]
}
