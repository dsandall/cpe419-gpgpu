	.file	"main.c"
	.text
	.globl	simple                          # -- Begin function simple
	.p2align	4
	.type	simple,@function
simple:                                 # @simple
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	$0, -32(%rbp)
.LBB0_1:                                # =>This Inner Loop Header: Depth=1
	cmpl	$20, -32(%rbp)
	jge	.LBB0_4
# %bb.2:                                #   in Loop: Header=BB0_1 Depth=1
	movq	-8(%rbp), %rax
	movslq	-32(%rbp), %rcx
	movb	(%rax,%rcx), %dl
	movq	-16(%rbp), %rax
	movslq	-32(%rbp), %rcx
	addb	(%rax,%rcx), %dl
	movq	-24(%rbp), %rax
	movslq	-32(%rbp), %rcx
	movb	%dl, (%rax,%rcx)
# %bb.3:                                #   in Loop: Header=BB0_1 Depth=1
	movl	-32(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -32(%rbp)
	jmp	.LBB0_1
.LBB0_4:
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end0:
	.size	simple, .Lfunc_end0-simple
	.cfi_endproc
                                        # -- End function
	.globl	matmul                          # -- Begin function matmul
	.p2align	4
	.type	matmul,@function
matmul:                                 # @matmul
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movl	$0, -40(%rbp)
.LBB1_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_3 Depth 2
                                        #       Child Loop BB1_5 Depth 3
	cmpl	$20, -40(%rbp)
	jge	.LBB1_12
# %bb.2:                                #   in Loop: Header=BB1_1 Depth=1
	movl	$0, -44(%rbp)
.LBB1_3:                                #   Parent Loop BB1_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_5 Depth 3
	cmpl	$30, -44(%rbp)
	jge	.LBB1_10
# %bb.4:                                #   in Loop: Header=BB1_3 Depth=2
	movl	$0, -48(%rbp)
.LBB1_5:                                #   Parent Loop BB1_1 Depth=1
                                        #     Parent Loop BB1_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	cmpl	$40, -48(%rbp)
	jge	.LBB1_8
# %bb.6:                                #   in Loop: Header=BB1_5 Depth=3
	movq	-8(%rbp), %rax
	movslq	-40(%rbp), %rcx
	imulq	$40, %rcx, %rcx
	addq	%rcx, %rax
	movslq	-48(%rbp), %rcx
	movb	(%rax,%rcx), %al
	movq	-16(%rbp), %rcx
	movslq	-48(%rbp), %rdx
	imulq	$30, %rdx, %rdx
	addq	%rdx, %rcx
	movslq	-44(%rbp), %rdx
	mulb	(%rcx,%rdx)
	movb	%al, %dl
	movq	-24(%rbp), %rax
	movslq	-40(%rbp), %rcx
	imulq	$30, %rcx, %rcx
	addq	%rcx, %rax
	movslq	-44(%rbp), %rcx
	addb	(%rax,%rcx), %dl
	movb	%dl, (%rax,%rcx)
# %bb.7:                                #   in Loop: Header=BB1_5 Depth=3
	movl	-48(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -48(%rbp)
	jmp	.LBB1_5
.LBB1_8:                                #   in Loop: Header=BB1_3 Depth=2
	jmp	.LBB1_9
.LBB1_9:                                #   in Loop: Header=BB1_3 Depth=2
	movl	-44(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -44(%rbp)
	jmp	.LBB1_3
.LBB1_10:                               #   in Loop: Header=BB1_1 Depth=1
	jmp	.LBB1_11
.LBB1_11:                               #   in Loop: Header=BB1_1 Depth=1
	movl	-40(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -40(%rbp)
	jmp	.LBB1_1
.LBB1_12:
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end1:
	.size	matmul, .Lfunc_end1-matmul
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 21.1.6"
	.section	".note.GNU-stack","",@progbits
	.addrsig
