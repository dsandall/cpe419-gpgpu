module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @simple(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c200 = arith.constant 200 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c200 step %c1 {
      %0 = memref.load %arg0[%arg3] : memref<?xi8>
      %1 = arith.extsi %0 : i8 to i32
      %2 = memref.load %arg1[%arg3] : memref<?xi8>
      %3 = arith.extsi %2 : i8 to i32
      %4 = arith.addi %1, %3 : i32
      %5 = arith.trunci %4 : i32 to i8
      memref.store %5, %arg2[%arg3] : memref<?xi8>
    }
    return
  }
}
