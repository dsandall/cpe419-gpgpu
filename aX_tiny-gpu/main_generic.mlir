"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8>):
    %0 = "arith.constant"() {value = 200 : index} : () -> index
    %1 = "arith.constant"() {value = 0 : index} : () -> index
    %2 = "arith.constant"() {value = 1 : index} : () -> index
    "scf.for"(%1, %0, %2) ({
    ^bb0(%arg3: index):
      %3 = "memref.load"(%arg0, %arg3) : (memref<?xi8>, index) -> i8
      %4 = "arith.extsi"(%3) : (i8) -> i32
      %5 = "memref.load"(%arg1, %arg3) : (memref<?xi8>, index) -> i8
      %6 = "arith.extsi"(%5) : (i8) -> i32
      %7 = "arith.addi"(%4, %6) : (i32, i32) -> i32
      %8 = "arith.trunci"(%7) : (i32) -> i8
      "memref.store"(%8, %arg2, %arg3) : (i8, memref<?xi8>, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {function_type = (memref<?xi8>, memref<?xi8>, memref<?xi8>) -> (), llvm.linkage = #llvm.linkage<external>, sym_name = "simple"} : () -> ()
}) {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} : () -> ()

