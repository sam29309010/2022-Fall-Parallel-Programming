	.file	"pi.c"
	.text
.Ltext0:
	.file 0 "/home/sam/Desktop/2022-Fall-Parallel-Programming/HW2/part1" "pi.c"
	.p2align 4
	.globl	single_thread_estimation_SIMD
	.type	single_thread_estimation_SIMD, @function
single_thread_estimation_SIMD:
.LVL0:
.LFB5700:
	.file 1 "pi.c"
	.loc 1 33 50 view -0
	.cfi_startproc
	.loc 1 33 50 is_stmt 0 view .LVU1
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r14
	pushq	%r13
	.cfi_offset 14, -24
	.cfi_offset 13, -32
	.loc 1 36 28 view .LVU2
	leaq	7(%rdi), %r13
	.loc 1 33 50 view .LVU3
	pushq	%r12
	pushq	%rbx
	.cfi_offset 12, -40
	.cfi_offset 3, -48
	movq	%rdi, %rbx
	andq	$-32, %rsp
	subq	$160, %rsp
	.loc 1 33 50 view .LVU4
	movq	%fs:40, %rax
	movq	%rax, 152(%rsp)
	xorl	%eax, %eax
	.loc 1 34 5 is_stmt 1 view .LVU5
	.loc 1 36 5 view .LVU6
.LVL1:
	.loc 1 36 28 is_stmt 0 view .LVU7
	testq	%rdi, %rdi
	.loc 1 51 5 view .LVU8
	leaq	64(%rsp), %r12
	.loc 1 36 28 view .LVU9
	cmovns	%rdi, %r13
	.loc 1 49 20 view .LVU10
	call	rand@PLT
.LVL2:
	.loc 1 49 20 view .LVU11
	movl	%eax, %r14d
	.loc 1 36 28 view .LVU12
	sarq	$3, %r13
.LVL3:
	.loc 1 38 5 is_stmt 1 view .LVU13
	.loc 1 39 5 view .LVU14
	.loc 1 40 5 view .LVU15
	.loc 1 42 5 view .LVU16
	.loc 1 43 5 view .LVU17
	.loc 1 44 5 view .LVU18
	.loc 1 45 5 view .LVU19
	.loc 1 47 5 view .LVU20
	.loc 1 49 5 view .LVU21
	.loc 1 50 5 view .LVU22
	.loc 1 50 20 is_stmt 0 view .LVU23
	call	rand@PLT
.LVL4:
	.loc 1 51 5 is_stmt 1 view .LVU24
	.loc 1 49 20 is_stmt 0 view .LVU25
	movslq	%r14d, %rdi
.LVL5:
	.loc 1 51 5 view .LVU26
	movq	%r12, %rdx
	.loc 1 50 20 view .LVU27
	movslq	%eax, %rsi
.LVL6:
	.loc 1 51 5 view .LVU28
	call	avx_xorshift128plus_init@PLT
.LVL7:
	.loc 1 54 5 is_stmt 1 view .LVU29
.LBB48:
	.loc 1 54 10 view .LVU30
	.loc 1 54 36 view .LVU31
	cmpq	$7, %rbx
	jle	.L5
	.loc 1 54 24 is_stmt 0 view .LVU32
	xorl	%ebx, %ebx
.LVL8:
	.loc 1 54 24 view .LVU33
.LBE48:
	.loc 1 43 17 view .LVU34
	vpxor	%xmm1, %xmm1, %xmm1
.LVL9:
	.p2align 4,,10
	.p2align 3
.L3:
.LBB85:
	.loc 1 56 19 discriminator 3 view .LVU35
	movq	%r12, %rdi
	vmovdqa	%ymm1, (%rsp)
.LVL10:
	.loc 1 56 9 is_stmt 1 discriminator 3 view .LVU36
	.loc 1 56 19 is_stmt 0 discriminator 3 view .LVU37
	vzeroupper
.LVL11:
	.loc 1 56 19 discriminator 3 view .LVU38
	call	avx_xorshift128plus@PLT
.LVL12:
	.loc 1 57 9 is_stmt 1 discriminator 3 view .LVU39
.LBB49:
.LBI49:
	.file 2 "/usr/lib/gcc/x86_64-linux-gnu/11/include/avxintrin.h"
	.loc 2 461 1 discriminator 3 view .LVU40
.LBB50:
	.loc 2 463 3 discriminator 3 view .LVU41
.LBE50:
.LBE49:
	.loc 1 61 19 is_stmt 0 discriminator 3 view .LVU42
	movq	%r12, %rdi
.LBB52:
.LBB51:
	.loc 2 463 18 discriminator 3 view .LVU43
	vcvtdq2ps	%ymm0, %ymm0
.LVL13:
	.loc 2 463 18 discriminator 3 view .LVU44
.LBE51:
.LBE52:
	.loc 1 58 9 is_stmt 1 discriminator 3 view .LVU45
.LBB53:
.LBI53:
	.loc 2 241 1 discriminator 3 view .LVU46
.LBB54:
	.loc 2 243 3 discriminator 3 view .LVU47
	.loc 2 243 10 is_stmt 0 discriminator 3 view .LVU48
	vmulps	.LC0(%rip), %ymm0, %ymm0
.LVL14:
	.loc 2 243 10 discriminator 3 view .LVU49
.LBE54:
.LBE53:
	.loc 1 59 9 is_stmt 1 discriminator 3 view .LVU50
.LBB55:
.LBI55:
	.loc 2 318 1 discriminator 3 view .LVU51
.LBB56:
	.loc 2 320 3 discriminator 3 view .LVU52
	.loc 2 320 3 is_stmt 0 discriminator 3 view .LVU53
.LBE56:
.LBE55:
	.loc 1 61 9 is_stmt 1 discriminator 3 view .LVU54
.LBB58:
.LBB57:
	.loc 2 320 10 is_stmt 0 discriminator 3 view .LVU55
	vmulps	%ymm0, %ymm0, %ymm2
	vmovaps	%ymm2, 32(%rsp)
.LVL15:
	.loc 2 320 10 discriminator 3 view .LVU56
.LBE57:
.LBE58:
	.loc 1 61 19 discriminator 3 view .LVU57
	vzeroupper
	.loc 1 54 56 discriminator 3 view .LVU58
	addq	$1, %rbx
.LVL16:
	.loc 1 61 19 discriminator 3 view .LVU59
	call	avx_xorshift128plus@PLT
.LVL17:
	.loc 1 62 9 is_stmt 1 discriminator 3 view .LVU60
.LBB59:
.LBI59:
	.loc 2 461 1 discriminator 3 view .LVU61
.LBB60:
	.loc 2 463 3 discriminator 3 view .LVU62
.LBE60:
.LBE59:
.LBB62:
.LBB63:
	.file 3 "/usr/lib/gcc/x86_64-linux-gnu/11/include/avx2intrin.h"
	.loc 3 121 10 is_stmt 0 discriminator 3 view .LVU63
	vmovdqa	(%rsp), %ymm1
.LBE63:
.LBE62:
.LBB65:
.LBB61:
	.loc 2 463 18 discriminator 3 view .LVU64
	vcvtdq2ps	%ymm0, %ymm0
.LVL18:
	.loc 2 463 18 discriminator 3 view .LVU65
.LBE61:
.LBE65:
	.loc 1 63 9 is_stmt 1 discriminator 3 view .LVU66
.LBB66:
.LBI66:
	.loc 2 241 1 discriminator 3 view .LVU67
.LBB67:
	.loc 2 243 3 discriminator 3 view .LVU68
	.loc 2 243 10 is_stmt 0 discriminator 3 view .LVU69
	vmulps	.LC0(%rip), %ymm0, %ymm0
.LVL19:
	.loc 2 243 10 discriminator 3 view .LVU70
.LBE67:
.LBE66:
	.loc 1 64 9 is_stmt 1 discriminator 3 view .LVU71
.LBB68:
.LBI68:
	.loc 2 318 1 discriminator 3 view .LVU72
.LBB69:
	.loc 2 320 3 discriminator 3 view .LVU73
	.loc 2 320 3 is_stmt 0 discriminator 3 view .LVU74
.LBE69:
.LBE68:
	.loc 1 67 9 is_stmt 1 discriminator 3 view .LVU75
.LBB71:
.LBI71:
	.loc 2 147 1 discriminator 3 view .LVU76
.LBB72:
	.loc 2 149 3 discriminator 3 view .LVU77
	.loc 2 149 3 is_stmt 0 discriminator 3 view .LVU78
.LBE72:
.LBE71:
	.loc 1 71 9 is_stmt 1 discriminator 3 view .LVU79
.LBB74:
.LBI74:
	.loc 2 404 1 discriminator 3 view .LVU80
.LBB75:
	.loc 2 406 3 discriminator 3 view .LVU81
.LBE75:
.LBE74:
.LBB77:
.LBB70:
	.loc 2 320 10 is_stmt 0 discriminator 3 view .LVU82
	vmulps	%ymm0, %ymm0, %ymm0
.LVL20:
	.loc 2 320 10 discriminator 3 view .LVU83
.LBE70:
.LBE77:
.LBB78:
.LBB73:
	.loc 2 149 10 discriminator 3 view .LVU84
	vaddps	32(%rsp), %ymm0, %ymm0
.LVL21:
	.loc 2 149 10 discriminator 3 view .LVU85
.LBE73:
.LBE78:
.LBB79:
.LBB76:
	.loc 2 406 19 discriminator 3 view .LVU86
	vcmpps	$2, .LC1(%rip), %ymm0, %ymm0
.LVL22:
	.loc 2 406 19 discriminator 3 view .LVU87
.LBE76:
.LBE79:
	.loc 1 72 9 is_stmt 1 discriminator 3 view .LVU88
.LBB80:
.LBI80:
	.loc 2 172 1 discriminator 3 view .LVU89
.LBB81:
	.loc 2 174 3 discriminator 3 view .LVU90
	.loc 2 174 19 is_stmt 0 discriminator 3 view .LVU91
	vandps	.LC1(%rip), %ymm0, %ymm0
.LVL23:
	.loc 2 174 19 discriminator 3 view .LVU92
.LBE81:
.LBE80:
	.loc 1 73 9 is_stmt 1 discriminator 3 view .LVU93
.LBB82:
.LBI82:
	.loc 2 473 1 discriminator 3 view .LVU94
.LBB83:
	.loc 2 475 3 discriminator 3 view .LVU95
	.loc 2 475 19 is_stmt 0 discriminator 3 view .LVU96
	vcvtps2dq	%ymm0, %ymm0
.LVL24:
	.loc 2 475 19 discriminator 3 view .LVU97
.LBE83:
.LBE82:
	.loc 1 74 9 is_stmt 1 discriminator 3 view .LVU98
.LBB84:
.LBI62:
	.loc 3 119 1 discriminator 3 view .LVU99
.LBB64:
	.loc 3 121 3 discriminator 3 view .LVU100
	.loc 3 121 10 is_stmt 0 discriminator 3 view .LVU101
	vpaddd	%ymm1, %ymm0, %ymm1
.LVL25:
	.loc 3 121 10 discriminator 3 view .LVU102
.LBE64:
.LBE84:
	.loc 1 54 56 is_stmt 1 discriminator 3 view .LVU103
	.loc 1 54 36 discriminator 3 view .LVU104
	cmpq	%rbx, %r13
	jg	.L3
.LVL26:
.L2:
	.loc 1 54 36 is_stmt 0 discriminator 3 view .LVU105
.LBE85:
	.loc 1 78 5 is_stmt 1 view .LVU106
	.loc 1 79 5 view .LVU107
.LBB86:
	.loc 1 79 10 view .LVU108
	.loc 1 79 20 view .LVU109
	.loc 1 80 9 view .LVU110
	.loc 1 80 23 is_stmt 0 view .LVU111
	vextracti128	$0x1, %ymm1, %xmm0
	vpmovsxdq	%xmm1, %ymm1
.LBE86:
	.loc 1 83 5 view .LVU112
	leaq	total_hit_mutex(%rip), %r12
.LBB87:
	.loc 1 80 23 view .LVU113
	vpmovsxdq	%xmm0, %ymm0
.LBE87:
	.loc 1 83 5 view .LVU114
	movq	%r12, %rdi
.LBB88:
	.loc 1 80 13 view .LVU115
	vpaddq	%ymm1, %ymm0, %ymm1
	.loc 1 79 25 is_stmt 1 view .LVU116
	.loc 1 79 20 view .LVU117
	vmovdqa	%xmm1, %xmm0
	vextracti128	$0x1, %ymm1, %xmm1
	vpaddq	%xmm1, %xmm0, %xmm0
	vmovdqa	%xmm0, 32(%rsp)
.LBE88:
	.loc 1 83 5 view .LVU118
	vzeroupper
	call	pthread_mutex_lock@PLT
.LVL27:
	.loc 1 84 5 view .LVU119
	.loc 1 84 15 is_stmt 0 view .LVU120
	vmovdqa	32(%rsp), %xmm0
	.loc 1 85 5 view .LVU121
	movq	%r12, %rdi
	.loc 1 84 15 view .LVU122
	vpsrldq	$8, %xmm0, %xmm1
	vpaddq	%xmm1, %xmm0, %xmm0
	vmovq	%xmm0, %rax
	addq	%rax, total_hit(%rip)
	.loc 1 85 5 is_stmt 1 view .LVU123
	call	pthread_mutex_unlock@PLT
.LVL28:
	.loc 1 86 5 view .LVU124
	.loc 1 87 1 is_stmt 0 view .LVU125
	movq	152(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L9
	leaq	-32(%rbp), %rsp
	xorl	%eax, %eax
	popq	%rbx
	popq	%r12
	popq	%r13
.LVL29:
	.loc 1 87 1 view .LVU126
	popq	%r14
.LVL30:
	.loc 1 87 1 view .LVU127
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.LVL31:
	.p2align 4,,10
	.p2align 3
.L5:
	.cfi_restore_state
	.loc 1 43 17 view .LVU128
	vpxor	%xmm1, %xmm1, %xmm1
	jmp	.L2
.LVL32:
.L9:
	.loc 1 87 1 view .LVU129
	call	__stack_chk_fail@PLT
.LVL33:
	.cfi_endproc
.LFE5700:
	.size	single_thread_estimation_SIMD, .-single_thread_estimation_SIMD
	.p2align 4
	.globl	single_thread_estimation_serial
	.type	single_thread_estimation_serial, @function
single_thread_estimation_serial:
.LVL34:
.LFB5699:
	.loc 1 13 52 is_stmt 1 view -0
	.cfi_startproc
	.loc 1 13 52 is_stmt 0 view .LVU131
	endbr64
	pushq	%r13
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	movq	%rdi, %r13
	.loc 1 14 25 view .LVU132
	xorl	%edi, %edi
.LVL35:
	.loc 1 13 52 view .LVU133
	pushq	%r12
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
	pushq	%rbp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	subq	$40, %rsp
	.cfi_def_cfa_offset 80
	.loc 1 13 52 view .LVU134
	movq	%fs:40, %rax
	movq	%rax, 24(%rsp)
	xorl	%eax, %eax
	.loc 1 14 5 is_stmt 1 view .LVU135
	.loc 1 14 25 is_stmt 0 view .LVU136
	call	time@PLT
.LVL36:
	.loc 1 14 18 view .LVU137
	movl	%eax, 20(%rsp)
	.loc 1 16 5 is_stmt 1 view .LVU138
.LVL37:
	.loc 1 17 5 view .LVU139
	.loc 1 18 5 view .LVU140
.LBB89:
	.loc 1 18 10 view .LVU141
	.loc 1 18 36 view .LVU142
	testq	%r13, %r13
	jle	.L16
	.loc 1 18 24 is_stmt 0 view .LVU143
	xorl	%ebx, %ebx
.LBE89:
	.loc 1 16 19 view .LVU144
	xorl	%ebp, %ebp
	leaq	20(%rsp), %r12
.LVL38:
	.p2align 4,,10
	.p2align 3
.L14:
.LBB90:
	.loc 1 19 9 is_stmt 1 view .LVU145
	.loc 1 19 23 is_stmt 0 view .LVU146
	movq	%r12, %rdi
	call	rand_r@PLT
.LVL39:
	.loc 1 19 14 view .LVU147
	vxorpd	%xmm2, %xmm2, %xmm2
	.loc 1 20 23 view .LVU148
	movq	%r12, %rdi
	.loc 1 19 14 view .LVU149
	vcvtsi2sdl	%eax, %xmm2, %xmm0
	.loc 1 19 11 view .LVU150
	vdivsd	.LC2(%rip), %xmm0, %xmm0
	vmovsd	%xmm0, 8(%rsp)
.LVL40:
	.loc 1 20 9 is_stmt 1 view .LVU151
	.loc 1 20 23 is_stmt 0 view .LVU152
	call	rand_r@PLT
.LVL41:
	.loc 1 21 18 view .LVU153
	vmovsd	8(%rsp), %xmm0
	.loc 1 20 14 view .LVU154
	vxorpd	%xmm2, %xmm2, %xmm2
	.loc 1 23 16 view .LVU155
	vmovsd	.LC3(%rip), %xmm3
	.loc 1 20 14 view .LVU156
	vcvtsi2sdl	%eax, %xmm2, %xmm1
	.loc 1 20 11 view .LVU157
	vdivsd	.LC2(%rip), %xmm1, %xmm1
.LVL42:
	.loc 1 21 9 is_stmt 1 view .LVU158
	.loc 1 22 9 view .LVU159
	.loc 1 21 26 is_stmt 0 view .LVU160
	vmulsd	%xmm1, %xmm1, %xmm1
.LVL43:
	.loc 1 21 18 view .LVU161
	vmulsd	%xmm0, %xmm0, %xmm0
	.loc 1 21 14 view .LVU162
	vaddsd	%xmm1, %xmm0, %xmm0
.LVL44:
	.loc 1 23 16 view .LVU163
	vcomisd	%xmm0, %xmm3
	sbbq	$-1, %rbp
.LVL45:
	.loc 1 18 51 is_stmt 1 view .LVU164
	addq	$1, %rbx
.LVL46:
	.loc 1 18 36 view .LVU165
	cmpq	%rbx, %r13
	jne	.L14
.LVL47:
.L11:
	.loc 1 18 36 is_stmt 0 view .LVU166
.LBE90:
	.loc 1 26 5 is_stmt 1 view .LVU167
	leaq	total_hit_mutex(%rip), %r12
	movq	%r12, %rdi
	call	pthread_mutex_lock@PLT
.LVL48:
	.loc 1 27 5 view .LVU168
	.loc 1 28 5 is_stmt 0 view .LVU169
	movq	%r12, %rdi
	.loc 1 27 15 view .LVU170
	addq	%rbp, total_hit(%rip)
	.loc 1 28 5 is_stmt 1 view .LVU171
	call	pthread_mutex_unlock@PLT
.LVL49:
	.loc 1 29 5 view .LVU172
	.loc 1 30 1 is_stmt 0 view .LVU173
	movq	24(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L20
	addq	$40, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	xorl	%eax, %eax
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
	popq	%r13
	.cfi_def_cfa_offset 8
.LVL50:
	.loc 1 30 1 view .LVU174
	ret
.LVL51:
	.p2align 4,,10
	.p2align 3
.L16:
	.cfi_restore_state
	.loc 1 16 19 view .LVU175
	xorl	%ebp, %ebp
	jmp	.L11
.LVL52:
.L20:
	.loc 1 30 1 view .LVU176
	call	__stack_chk_fail@PLT
.LVL53:
	.cfi_endproc
.LFE5699:
	.size	single_thread_estimation_serial, .-single_thread_estimation_serial
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align 8
.LC4:
	.string	"Usage of command: ./pi.out #Thread #Tosses"
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC5:
	.string	"e.g. ./pi.out 8 1000000000"
.LC6:
	.string	"%lf\n"
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.globl	main
	.type	main, @function
main:
.LVL54:
.LFB5701:
	.loc 1 89 32 is_stmt 1 view -0
	.cfi_startproc
	.loc 1 89 32 is_stmt 0 view .LVU178
	endbr64
	.loc 1 91 5 is_stmt 1 view .LVU179
	.loc 1 89 32 is_stmt 0 view .LVU180
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$24, %rsp
	.cfi_def_cfa_offset 80
	.loc 1 91 8 view .LVU181
	cmpl	$3, %edi
	je	.L22
	.loc 1 92 9 is_stmt 1 view .LVU182
.LVL55:
.LBB91:
.LBI91:
	.file 4 "/usr/include/x86_64-linux-gnu/bits/stdio2.h"
	.loc 4 110 1 view .LVU183
.LBB92:
	.loc 4 112 3 view .LVU184
	.loc 4 112 10 is_stmt 0 view .LVU185
	leaq	.LC4(%rip), %rdi
.LVL56:
	.loc 4 112 10 view .LVU186
	call	puts@PLT
.LVL57:
	.loc 4 112 10 view .LVU187
.LBE92:
.LBE91:
	.loc 1 93 9 is_stmt 1 view .LVU188
.LBB93:
.LBI93:
	.loc 4 110 1 view .LVU189
.LBB94:
	.loc 4 112 3 view .LVU190
	.loc 4 112 10 is_stmt 0 view .LVU191
	leaq	.LC5(%rip), %rdi
	call	puts@PLT
.LVL58:
	.loc 4 112 10 view .LVU192
.LBE94:
.LBE93:
	.loc 1 94 9 is_stmt 1 view .LVU193
	.loc 1 94 16 is_stmt 0 view .LVU194
	movl	$1, %eax
.L21:
	.loc 1 120 1 view .LVU195
	addq	$24, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
.LVL59:
.L22:
	.cfi_restore_state
.LBB95:
.LBB96:
	.file 5 "/usr/include/stdlib.h"
	.loc 5 364 16 view .LVU196
	movq	8(%rsi), %rdi
.LVL60:
	.loc 5 364 16 view .LVU197
	movq	%rsi, %rbx
.LBE96:
.LBE95:
	.loc 1 98 5 is_stmt 1 view .LVU198
.LVL61:
.LBB99:
.LBI95:
	.loc 5 362 1 view .LVU199
.LBB97:
	.loc 5 364 3 view .LVU200
	.loc 5 364 16 is_stmt 0 view .LVU201
	movl	$10, %edx
	xorl	%esi, %esi
.LVL62:
	.loc 5 364 16 view .LVU202
	call	strtol@PLT
.LVL63:
	.loc 5 364 16 view .LVU203
.LBE97:
.LBE99:
.LBB100:
.LBB101:
	movq	16(%rbx), %rdi
	xorl	%esi, %esi
	movl	$10, %edx
.LBE101:
.LBE100:
.LBB104:
.LBB98:
	movq	%rax, %rbp
.LVL64:
	.loc 5 364 16 view .LVU204
.LBE98:
.LBE104:
	.loc 1 99 5 is_stmt 1 view .LVU205
.LBB105:
.LBI100:
	.loc 5 362 1 view .LVU206
.LBB102:
	.loc 5 364 3 view .LVU207
	.loc 5 364 16 is_stmt 0 view .LVU208
	call	strtol@PLT
.LVL65:
	.loc 5 364 16 view .LVU209
.LBE102:
.LBE105:
	.loc 1 100 50 view .LVU210
	movslq	%ebp, %rdi
.LBB106:
.LBB103:
	.loc 5 364 16 view .LVU211
	movq	%rax, 8(%rsp)
.LVL66:
	.loc 5 364 16 view .LVU212
.LBE103:
.LBE106:
	.loc 1 100 5 is_stmt 1 view .LVU213
	.loc 1 99 19 is_stmt 0 view .LVU214
	cltq
.LVL67:
	.loc 1 100 19 view .LVU215
	cqto
	idivq	%rdi
.LVL68:
	.loc 1 104 47 view .LVU216
	salq	$3, %rdi
	.loc 1 100 19 view .LVU217
	movq	%rax, %r12
.LVL69:
	.loc 1 101 5 is_stmt 1 view .LVU218
	.loc 1 104 5 view .LVU219
	.loc 1 104 47 is_stmt 0 view .LVU220
	call	malloc@PLT
.LVL70:
	.loc 1 105 5 view .LVU221
	xorl	%esi, %esi
	leaq	total_hit_mutex(%rip), %rdi
	.loc 1 104 47 view .LVU222
	movq	%rax, %r14
.LVL71:
	.loc 1 105 5 is_stmt 1 view .LVU223
	call	pthread_mutex_init@PLT
.LVL72:
	.loc 1 106 5 view .LVU224
.LBB107:
	.loc 1 106 10 view .LVU225
	.loc 1 106 20 view .LVU226
	testl	%ebp, %ebp
	jle	.L27
	.loc 1 106 20 is_stmt 0 view .LVU227
	leal	-1(%rbp), %eax
	movq	%r14, %rbx
.LVL73:
	.loc 1 106 20 view .LVU228
	leaq	single_thread_estimation_SIMD(%rip), %r13
	.loc 1 107 9 view .LVU229
	movq	%r14, %r15
	leaq	8(%r14,%rax,8), %rbp
.LVL74:
	.p2align 4,,10
	.p2align 3
.L25:
	.loc 1 107 9 is_stmt 1 discriminator 3 view .LVU230
	movq	%r15, %rdi
	movq	%r12, %rcx
	movq	%r13, %rdx
	xorl	%esi, %esi
	call	pthread_create@PLT
.LVL75:
	.loc 1 106 34 discriminator 3 view .LVU231
	.loc 1 106 20 discriminator 3 view .LVU232
	addq	$8, %r15
.LVL76:
	.loc 1 106 20 is_stmt 0 discriminator 3 view .LVU233
	cmpq	%rbp, %r15
	jne	.L25
.LVL77:
	.p2align 4,,10
	.p2align 3
.L26:
	.loc 1 106 20 discriminator 3 view .LVU234
.LBE107:
.LBB108:
	.loc 1 112 9 is_stmt 1 discriminator 3 view .LVU235
	movq	(%rbx), %rdi
	xorl	%esi, %esi
	.loc 1 111 20 is_stmt 0 discriminator 3 view .LVU236
	addq	$8, %rbx
.LVL78:
	.loc 1 112 9 discriminator 3 view .LVU237
	call	pthread_join@PLT
.LVL79:
	.loc 1 111 34 is_stmt 1 discriminator 3 view .LVU238
	.loc 1 111 20 discriminator 3 view .LVU239
	cmpq	%rbp, %rbx
	jne	.L26
.LVL80:
.L27:
	.loc 1 111 20 is_stmt 0 discriminator 3 view .LVU240
.LBE108:
	.loc 1 115 5 is_stmt 1 view .LVU241
	.loc 1 116 5 view .LVU242
.LBB109:
.LBI109:
	.loc 4 110 1 view .LVU243
.LBB110:
	.loc 4 112 3 view .LVU244
.LBE110:
.LBE109:
	.loc 1 115 16 is_stmt 0 view .LVU245
	movq	total_hit(%rip), %rax
	vxorps	%xmm1, %xmm1, %xmm1
.LBB114:
.LBB111:
	.loc 4 112 10 view .LVU246
	movl	$1, %edi
	leaq	.LC6(%rip), %rsi
.LBE111:
.LBE114:
	.loc 1 115 16 view .LVU247
	salq	$2, %rax
	.loc 1 115 28 view .LVU248
	vcvtsi2sdq	%rax, %xmm1, %xmm0
.LBB115:
.LBB112:
	.loc 4 112 10 view .LVU249
	movl	$1, %eax
.LBE112:
.LBE115:
	.loc 1 115 31 view .LVU250
	vcvtsi2sdl	8(%rsp), %xmm1, %xmm1
	.loc 1 115 12 view .LVU251
	vdivsd	%xmm1, %xmm0, %xmm0
.LBB116:
.LBB113:
	.loc 4 112 10 view .LVU252
	call	__printf_chk@PLT
.LVL81:
	.loc 4 112 10 view .LVU253
.LBE113:
.LBE116:
	.loc 1 118 5 is_stmt 1 view .LVU254
	leaq	total_hit_mutex(%rip), %rdi
	call	pthread_mutex_destroy@PLT
.LVL82:
	.loc 1 119 5 view .LVU255
	movq	%r14, %rdi
	call	free@PLT
.LVL83:
	xorl	%eax, %eax
	jmp	.L21
	.cfi_endproc
.LFE5701:
	.size	main, .-main
	.globl	total_hit
	.bss
	.align 8
	.type	total_hit, @object
	.size	total_hit, 8
total_hit:
	.zero	8
	.globl	total_hit_mutex
	.align 32
	.type	total_hit_mutex, @object
	.size	total_hit_mutex, 40
total_hit_mutex:
	.zero	40
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC0:
	.long	805306368
	.long	805306368
	.long	805306368
	.long	805306368
	.long	805306368
	.long	805306368
	.long	805306368
	.long	805306368
	.align 32
.LC1:
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.section	.rodata.cst8,"aM",@progbits,8
	.align 8
.LC2:
	.long	-4194304
	.long	1105199103
	.align 8
.LC3:
	.long	0
	.long	1072693248
	.text
.Letext0:
	.file 6 "/usr/lib/gcc/x86_64-linux-gnu/11/include/stddef.h"
	.file 7 "/usr/include/x86_64-linux-gnu/bits/types.h"
	.file 8 "/usr/include/x86_64-linux-gnu/bits/types/time_t.h"
	.file 9 "/usr/include/x86_64-linux-gnu/bits/thread-shared-types.h"
	.file 10 "/usr/include/x86_64-linux-gnu/bits/struct_mutex.h"
	.file 11 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h"
	.file 12 "/usr/include/x86_64-linux-gnu/bits/stdint-uintn.h"
	.file 13 "SIMDxorshift/include/simdxorshift128plus.h"
	.file 14 "/usr/include/pthread.h"
	.file 15 "/usr/include/time.h"
	.file 16 "<built-in>"
	.section	.debug_info,"",@progbits
.Ldebug_info0:
	.long	0x10b6
	.value	0x5
	.byte	0x1
	.byte	0x8
	.long	.Ldebug_abbrev0
	.uleb128 0x26
	.long	.LASF109
	.byte	0x1d
	.long	.LASF0
	.long	.LASF1
	.long	.LLRL70
	.quad	0
	.long	.Ldebug_line0
	.uleb128 0x8
	.byte	0x8
	.byte	0x4
	.long	.LASF2
	.uleb128 0x8
	.byte	0x8
	.byte	0x5
	.long	.LASF3
	.uleb128 0x27
	.byte	0x4
	.byte	0x5
	.string	"int"
	.uleb128 0x6
	.long	.LASF10
	.byte	0x6
	.byte	0xd1
	.byte	0x17
	.long	0x4b
	.uleb128 0x8
	.byte	0x8
	.byte	0x7
	.long	.LASF4
	.uleb128 0x8
	.byte	0x4
	.byte	0x7
	.long	.LASF5
	.uleb128 0x28
	.byte	0x8
	.uleb128 0x14
	.long	0x59
	.uleb128 0x8
	.byte	0x1
	.byte	0x8
	.long	.LASF6
	.uleb128 0x8
	.byte	0x2
	.byte	0x7
	.long	.LASF7
	.uleb128 0x8
	.byte	0x1
	.byte	0x6
	.long	.LASF8
	.uleb128 0x8
	.byte	0x2
	.byte	0x5
	.long	.LASF9
	.uleb128 0x1a
	.long	0x38
	.uleb128 0x6
	.long	.LASF11
	.byte	0x7
	.byte	0x2d
	.byte	0x1b
	.long	0x4b
	.uleb128 0x6
	.long	.LASF12
	.byte	0x7
	.byte	0xa0
	.byte	0x1a
	.long	0x31
	.uleb128 0x7
	.long	0x9e
	.uleb128 0x8
	.byte	0x1
	.byte	0x6
	.long	.LASF13
	.uleb128 0x1a
	.long	0x9e
	.uleb128 0x15
	.long	0x9e
	.long	0xba
	.uleb128 0x16
	.long	0x4b
	.byte	0x3
	.byte	0
	.uleb128 0x8
	.byte	0x8
	.byte	0x5
	.long	.LASF14
	.uleb128 0x6
	.long	.LASF15
	.byte	0x8
	.byte	0xa
	.byte	0x12
	.long	0x8d
	.uleb128 0x8
	.byte	0x8
	.byte	0x7
	.long	.LASF16
	.uleb128 0x1d
	.long	.LASF20
	.byte	0x10
	.byte	0x9
	.byte	0x33
	.byte	0x10
	.long	0xfc
	.uleb128 0x9
	.long	.LASF17
	.byte	0x9
	.byte	0x35
	.byte	0x23
	.long	0xfc
	.byte	0
	.uleb128 0x9
	.long	.LASF18
	.byte	0x9
	.byte	0x36
	.byte	0x23
	.long	0xfc
	.byte	0x8
	.byte	0
	.uleb128 0x7
	.long	0xd4
	.uleb128 0x6
	.long	.LASF19
	.byte	0x9
	.byte	0x37
	.byte	0x3
	.long	0xd4
	.uleb128 0x1d
	.long	.LASF21
	.byte	0x28
	.byte	0xa
	.byte	0x16
	.byte	0x8
	.long	0x183
	.uleb128 0x9
	.long	.LASF22
	.byte	0xa
	.byte	0x18
	.byte	0x7
	.long	0x38
	.byte	0
	.uleb128 0x9
	.long	.LASF23
	.byte	0xa
	.byte	0x19
	.byte	0x10
	.long	0x52
	.byte	0x4
	.uleb128 0x9
	.long	.LASF24
	.byte	0xa
	.byte	0x1a
	.byte	0x7
	.long	0x38
	.byte	0x8
	.uleb128 0x9
	.long	.LASF25
	.byte	0xa
	.byte	0x1c
	.byte	0x10
	.long	0x52
	.byte	0xc
	.uleb128 0x9
	.long	.LASF26
	.byte	0xa
	.byte	0x20
	.byte	0x7
	.long	0x38
	.byte	0x10
	.uleb128 0x9
	.long	.LASF27
	.byte	0xa
	.byte	0x22
	.byte	0x9
	.long	0x75
	.byte	0x14
	.uleb128 0x9
	.long	.LASF28
	.byte	0xa
	.byte	0x23
	.byte	0x9
	.long	0x75
	.byte	0x16
	.uleb128 0x9
	.long	.LASF29
	.byte	0xa
	.byte	0x24
	.byte	0x14
	.long	0x101
	.byte	0x18
	.byte	0
	.uleb128 0x6
	.long	.LASF30
	.byte	0xb
	.byte	0x1b
	.byte	0x1b
	.long	0x4b
	.uleb128 0x20
	.byte	0x4
	.byte	0x20
	.long	0x1ad
	.uleb128 0xe
	.long	.LASF31
	.byte	0x22
	.byte	0x8
	.long	0xaa
	.uleb128 0xe
	.long	.LASF32
	.byte	0x23
	.byte	0x7
	.long	0x38
	.byte	0
	.uleb128 0x6
	.long	.LASF33
	.byte	0xb
	.byte	0x24
	.byte	0x3
	.long	0x18f
	.uleb128 0x1a
	.long	0x1ad
	.uleb128 0x29
	.long	.LASF34
	.byte	0x38
	.byte	0xb
	.byte	0x38
	.byte	0x7
	.long	0x1e2
	.uleb128 0xe
	.long	.LASF31
	.byte	0x3a
	.byte	0x8
	.long	0x1e2
	.uleb128 0xe
	.long	.LASF32
	.byte	0x3b
	.byte	0xc
	.long	0x31
	.byte	0
	.uleb128 0x15
	.long	0x9e
	.long	0x1f2
	.uleb128 0x16
	.long	0x4b
	.byte	0x37
	.byte	0
	.uleb128 0x6
	.long	.LASF34
	.byte	0xb
	.byte	0x3e
	.byte	0x1e
	.long	0x1be
	.uleb128 0x1a
	.long	0x1f2
	.uleb128 0x20
	.byte	0x28
	.byte	0x43
	.long	0x22c
	.uleb128 0xe
	.long	.LASF35
	.byte	0x45
	.byte	0x1c
	.long	0x10d
	.uleb128 0xe
	.long	.LASF31
	.byte	0x46
	.byte	0x8
	.long	0x22c
	.uleb128 0xe
	.long	.LASF32
	.byte	0x47
	.byte	0xc
	.long	0x31
	.byte	0
	.uleb128 0x15
	.long	0x9e
	.long	0x23c
	.uleb128 0x16
	.long	0x4b
	.byte	0x27
	.byte	0
	.uleb128 0x6
	.long	.LASF36
	.byte	0xb
	.byte	0x48
	.byte	0x3
	.long	0x203
	.uleb128 0x7
	.long	0xa5
	.uleb128 0x14
	.long	0x248
	.uleb128 0x8
	.byte	0x10
	.byte	0x4
	.long	.LASF37
	.uleb128 0x8
	.byte	0x4
	.byte	0x4
	.long	.LASF38
	.uleb128 0x6
	.long	.LASF39
	.byte	0x2
	.byte	0x2a
	.byte	0xf
	.long	0x26c
	.uleb128 0x17
	.long	0x259
	.long	0x278
	.uleb128 0x18
	.byte	0x7
	.byte	0
	.uleb128 0x6
	.long	.LASF40
	.byte	0x2
	.byte	0x2d
	.byte	0xd
	.long	0x284
	.uleb128 0x17
	.long	0x38
	.long	0x290
	.uleb128 0x18
	.byte	0x7
	.byte	0
	.uleb128 0x6
	.long	.LASF41
	.byte	0x2
	.byte	0x2e
	.byte	0x16
	.long	0x29c
	.uleb128 0x17
	.long	0x52
	.long	0x2a8
	.uleb128 0x18
	.byte	0x7
	.byte	0
	.uleb128 0x6
	.long	.LASF42
	.byte	0x2
	.byte	0x37
	.byte	0xf
	.long	0x2b4
	.uleb128 0x17
	.long	0x259
	.long	0x2c0
	.uleb128 0x18
	.byte	0x7
	.byte	0
	.uleb128 0x6
	.long	.LASF43
	.byte	0x2
	.byte	0x39
	.byte	0x13
	.long	0x2cc
	.uleb128 0x17
	.long	0xba
	.long	0x2d8
	.uleb128 0x18
	.byte	0x3
	.byte	0
	.uleb128 0x6
	.long	.LASF44
	.byte	0xc
	.byte	0x1b
	.byte	0x14
	.long	0x81
	.uleb128 0x1d
	.long	.LASF45
	.byte	0x40
	.byte	0xd
	.byte	0x1e
	.byte	0x8
	.long	0x30c
	.uleb128 0x9
	.long	.LASF46
	.byte	0xd
	.byte	0x1f
	.byte	0xd
	.long	0x2c0
	.byte	0
	.uleb128 0x9
	.long	.LASF47
	.byte	0xd
	.byte	0x20
	.byte	0xd
	.long	0x2c0
	.byte	0x20
	.byte	0
	.uleb128 0x6
	.long	.LASF48
	.byte	0xd
	.byte	0x23
	.byte	0x2a
	.long	0x2e4
	.uleb128 0x21
	.long	.LASF49
	.byte	0x9
	.byte	0x11
	.long	0x23c
	.uleb128 0x9
	.byte	0x3
	.quad	total_hit_mutex
	.uleb128 0x21
	.long	.LASF50
	.byte	0xa
	.byte	0xf
	.long	0xba
	.uleb128 0x9
	.byte	0x3
	.quad	total_hit
	.uleb128 0x10
	.long	.LASF51
	.byte	0x4
	.byte	0x5f
	.byte	0xc
	.long	0x38
	.long	0x35e
	.uleb128 0x2
	.long	0x38
	.uleb128 0x2
	.long	0x248
	.uleb128 0x22
	.byte	0
	.uleb128 0x10
	.long	.LASF52
	.byte	0x5
	.byte	0xb1
	.byte	0x11
	.long	0x31
	.long	0x37e
	.uleb128 0x2
	.long	0x24d
	.uleb128 0x2
	.long	0x383
	.uleb128 0x2
	.long	0x38
	.byte	0
	.uleb128 0x7
	.long	0x99
	.uleb128 0x14
	.long	0x37e
	.uleb128 0x2a
	.long	.LASF59
	.byte	0x5
	.value	0x22b
	.byte	0xd
	.long	0x39b
	.uleb128 0x2
	.long	0x59
	.byte	0
	.uleb128 0x11
	.long	.LASF53
	.byte	0xe
	.value	0x312
	.byte	0xc
	.long	0x38
	.long	0x3b2
	.uleb128 0x2
	.long	0x3b2
	.byte	0
	.uleb128 0x7
	.long	0x23c
	.uleb128 0x10
	.long	.LASF54
	.byte	0xe
	.byte	0xdb
	.byte	0xc
	.long	0x38
	.long	0x3d2
	.uleb128 0x2
	.long	0x183
	.uleb128 0x2
	.long	0x3d2
	.byte	0
	.uleb128 0x7
	.long	0x59
	.uleb128 0x10
	.long	.LASF55
	.byte	0xe
	.byte	0xca
	.byte	0xc
	.long	0x38
	.long	0x3fc
	.uleb128 0x2
	.long	0x401
	.uleb128 0x2
	.long	0x40b
	.uleb128 0x2
	.long	0x410
	.uleb128 0x2
	.long	0x5b
	.byte	0
	.uleb128 0x7
	.long	0x183
	.uleb128 0x14
	.long	0x3fc
	.uleb128 0x7
	.long	0x1fe
	.uleb128 0x14
	.long	0x406
	.uleb128 0x7
	.long	0x415
	.uleb128 0x2b
	.long	0x59
	.long	0x424
	.uleb128 0x2
	.long	0x59
	.byte	0
	.uleb128 0x11
	.long	.LASF56
	.byte	0xe
	.value	0x30d
	.byte	0xc
	.long	0x38
	.long	0x440
	.uleb128 0x2
	.long	0x3b2
	.uleb128 0x2
	.long	0x440
	.byte	0
	.uleb128 0x7
	.long	0x1b9
	.uleb128 0x11
	.long	.LASF57
	.byte	0x5
	.value	0x21c
	.byte	0xe
	.long	0x59
	.long	0x45c
	.uleb128 0x2
	.long	0x3f
	.byte	0
	.uleb128 0x10
	.long	.LASF58
	.byte	0xd
	.byte	0x3d
	.byte	0x9
	.long	0x2c0
	.long	0x472
	.uleb128 0x2
	.long	0x472
	.byte	0
	.uleb128 0x7
	.long	0x30c
	.uleb128 0x2c
	.long	.LASF60
	.byte	0xd
	.byte	0x38
	.byte	0x6
	.long	0x493
	.uleb128 0x2
	.long	0x2d8
	.uleb128 0x2
	.long	0x2d8
	.uleb128 0x2
	.long	0x472
	.byte	0
	.uleb128 0x2d
	.long	.LASF110
	.byte	0x5
	.value	0x1c6
	.byte	0xc
	.long	0x38
	.uleb128 0x11
	.long	.LASF61
	.byte	0xe
	.value	0x343
	.byte	0xc
	.long	0x38
	.long	0x4b7
	.uleb128 0x2
	.long	0x3b2
	.byte	0
	.uleb128 0x11
	.long	.LASF62
	.byte	0xe
	.value	0x31a
	.byte	0xc
	.long	0x38
	.long	0x4ce
	.uleb128 0x2
	.long	0x3b2
	.byte	0
	.uleb128 0x11
	.long	.LASF63
	.byte	0x5
	.value	0x1cc
	.byte	0xc
	.long	0x38
	.long	0x4e5
	.uleb128 0x2
	.long	0x4e5
	.byte	0
	.uleb128 0x7
	.long	0x52
	.uleb128 0x10
	.long	.LASF64
	.byte	0xf
	.byte	0x4c
	.byte	0xf
	.long	0xc1
	.long	0x500
	.uleb128 0x2
	.long	0x500
	.byte	0
	.uleb128 0x7
	.long	0xc1
	.uleb128 0x1e
	.long	.LASF72
	.byte	0x59
	.byte	0x5
	.long	0x38
	.quad	.LFB5701
	.quad	.LFE5701-.LFB5701
	.uleb128 0x1
	.byte	0x9c
	.long	0x805
	.uleb128 0x1b
	.long	.LASF65
	.byte	0x59
	.byte	0xe
	.long	0x38
	.long	.LLST53
	.long	.LVUS53
	.uleb128 0x1b
	.long	.LASF66
	.byte	0x59
	.byte	0x1b
	.long	0x37e
	.long	.LLST54
	.long	.LVUS54
	.uleb128 0x5
	.long	.LASF67
	.byte	0x62
	.byte	0x9
	.long	0x38
	.long	.LLST55
	.long	.LVUS55
	.uleb128 0x5
	.long	.LASF68
	.byte	0x63
	.byte	0x13
	.long	0xba
	.long	.LLST56
	.long	.LVUS56
	.uleb128 0x5
	.long	.LASF69
	.byte	0x64
	.byte	0x13
	.long	0xba
	.long	.LLST57
	.long	.LVUS57
	.uleb128 0x5
	.long	.LASF70
	.byte	0x65
	.byte	0xc
	.long	0x2a
	.long	.LLST58
	.long	.LVUS58
	.uleb128 0x5
	.long	.LASF71
	.byte	0x68
	.byte	0x10
	.long	0x3fc
	.long	.LLST59
	.long	.LVUS59
	.uleb128 0x23
	.quad	.LBB107
	.quad	.LBE107-.LBB107
	.long	0x5f7
	.uleb128 0xf
	.string	"i"
	.byte	0x6a
	.byte	0xe
	.long	0x38
	.long	.LLST66
	.long	.LVUS66
	.uleb128 0xb
	.quad	.LVL75
	.long	0x3d7
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x2
	.byte	0x7f
	.sleb128 0
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x1
	.byte	0x30
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x51
	.uleb128 0x2
	.byte	0x7d
	.sleb128 0
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x52
	.uleb128 0x2
	.byte	0x7c
	.sleb128 0
	.byte	0
	.byte	0
	.uleb128 0x23
	.quad	.LBB108
	.quad	.LBE108-.LBB108
	.long	0x631
	.uleb128 0xf
	.string	"i"
	.byte	0x6f
	.byte	0xe
	.long	0x38
	.long	.LLST67
	.long	.LVUS67
	.uleb128 0xb
	.quad	.LVL79
	.long	0x3b7
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x1
	.byte	0x30
	.byte	0
	.byte	0
	.uleb128 0x12
	.long	0x1038
	.quad	.LBI91
	.byte	.LVU183
	.quad	.LBB91
	.quad	.LBE91-.LBB91
	.byte	0x5c
	.byte	0x9
	.long	0x67e
	.uleb128 0x4
	.long	0x1047
	.long	.LLST60
	.long	.LVUS60
	.uleb128 0xb
	.quad	.LVL57
	.long	0x1085
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x9
	.byte	0x3
	.quad	.LC4
	.byte	0
	.byte	0
	.uleb128 0x12
	.long	0x1038
	.quad	.LBI93
	.byte	.LVU189
	.quad	.LBB93
	.quad	.LBE93-.LBB93
	.byte	0x5d
	.byte	0x9
	.long	0x6cb
	.uleb128 0x4
	.long	0x1047
	.long	.LLST61
	.long	.LVUS61
	.uleb128 0xb
	.quad	.LVL58
	.long	0x1085
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x9
	.byte	0x3
	.quad	.LC5
	.byte	0
	.byte	0
	.uleb128 0xc
	.long	0x1018
	.quad	.LBI95
	.byte	.LVU199
	.long	.LLRL62
	.byte	0x62
	.byte	0x16
	.long	0x709
	.uleb128 0x4
	.long	0x102a
	.long	.LLST63
	.long	.LVUS63
	.uleb128 0xb
	.quad	.LVL63
	.long	0x35e
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x1
	.byte	0x30
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x51
	.uleb128 0x1
	.byte	0x3a
	.byte	0
	.byte	0
	.uleb128 0xc
	.long	0x1018
	.quad	.LBI100
	.byte	.LVU206
	.long	.LLRL64
	.byte	0x63
	.byte	0x20
	.long	0x747
	.uleb128 0x4
	.long	0x102a
	.long	.LLST65
	.long	.LVUS65
	.uleb128 0xb
	.quad	.LVL65
	.long	0x35e
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x1
	.byte	0x30
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x51
	.uleb128 0x1
	.byte	0x3a
	.byte	0
	.byte	0
	.uleb128 0xc
	.long	0x1038
	.quad	.LBI109
	.byte	.LVU243
	.long	.LLRL68
	.byte	0x74
	.byte	0x5
	.long	0x78d
	.uleb128 0x4
	.long	0x1047
	.long	.LLST69
	.long	.LVUS69
	.uleb128 0xb
	.quad	.LVL81
	.long	0x342
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x1
	.byte	0x31
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x9
	.byte	0x3
	.quad	.LC6
	.byte	0
	.byte	0
	.uleb128 0xa
	.quad	.LVL70
	.long	0x445
	.long	0x7ad
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x55
	.uleb128 0xa
	.byte	0x76
	.sleb128 0
	.byte	0x8
	.byte	0x20
	.byte	0x24
	.byte	0x8
	.byte	0x20
	.byte	0x26
	.byte	0x33
	.byte	0x24
	.byte	0
	.uleb128 0xa
	.quad	.LVL72
	.long	0x424
	.long	0x7d1
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x9
	.byte	0x3
	.quad	total_hit_mutex
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x1
	.byte	0x30
	.byte	0
	.uleb128 0xa
	.quad	.LVL82
	.long	0x39b
	.long	0x7f0
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x9
	.byte	0x3
	.quad	total_hit_mutex
	.byte	0
	.uleb128 0xb
	.quad	.LVL83
	.long	0x388
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x2
	.byte	0x7e
	.sleb128 0
	.byte	0
	.byte	0
	.uleb128 0x1e
	.long	.LASF73
	.byte	0x21
	.byte	0x7
	.long	0x59
	.quad	.LFB5700
	.quad	.LFE5700-.LFB5700
	.uleb128 0x1
	.byte	0x9c
	.long	0xc7d
	.uleb128 0x1b
	.long	.LASF74
	.byte	0x21
	.byte	0x2b
	.long	0x59
	.long	.LLST0
	.long	.LVUS0
	.uleb128 0x5
	.long	.LASF75
	.byte	0x22
	.byte	0xe
	.long	0xc7d
	.long	.LLST1
	.long	.LVUS1
	.uleb128 0xf
	.string	"hit"
	.byte	0x24
	.byte	0x13
	.long	0xba
	.long	.LLST2
	.long	.LVUS2
	.uleb128 0x5
	.long	.LASF76
	.byte	0x24
	.byte	0x1c
	.long	0xba
	.long	.LLST3
	.long	.LVUS3
	.uleb128 0x2e
	.long	.LASF111
	.byte	0x1
	.byte	0x26
	.byte	0x9
	.long	0xc8d
	.uleb128 0x5
	.long	.LASF77
	.byte	0x27
	.byte	0xd
	.long	0x2c0
	.long	.LLST4
	.long	.LVUS4
	.uleb128 0x5
	.long	.LASF78
	.byte	0x27
	.byte	0x16
	.long	0x2c0
	.long	.LLST5
	.long	.LVUS5
	.uleb128 0x5
	.long	.LASF79
	.byte	0x27
	.byte	0x1f
	.long	0x2c0
	.long	.LLST6
	.long	.LVUS6
	.uleb128 0x5
	.long	.LASF80
	.byte	0x27
	.byte	0x2a
	.long	0x2c0
	.long	.LLST7
	.long	.LVUS7
	.uleb128 0x5
	.long	.LASF81
	.byte	0x28
	.byte	0xc
	.long	0x2a8
	.long	.LLST8
	.long	.LVUS8
	.uleb128 0x5
	.long	.LASF82
	.byte	0x28
	.byte	0x16
	.long	0x2a8
	.long	.LLST9
	.long	.LVUS9
	.uleb128 0x5
	.long	.LASF83
	.byte	0x28
	.byte	0x20
	.long	0x2a8
	.long	.LLST10
	.long	.LVUS10
	.uleb128 0x5
	.long	.LASF84
	.byte	0x28
	.byte	0x2d
	.long	0x2a8
	.long	.LLST11
	.long	.LVUS11
	.uleb128 0x5
	.long	.LASF85
	.byte	0x28
	.byte	0x3c
	.long	0x2a8
	.long	.LLST12
	.long	.LVUS12
	.uleb128 0x5
	.long	.LASF86
	.byte	0x28
	.byte	0x48
	.long	0x2a8
	.long	.LLST13
	.long	.LVUS13
	.uleb128 0x5
	.long	.LASF87
	.byte	0x28
	.byte	0x54
	.long	0x2a8
	.long	.LLST14
	.long	.LVUS14
	.uleb128 0x24
	.long	.LASF90
	.byte	0x2f
	.byte	0x1f
	.long	0x30c
	.uleb128 0x3
	.byte	0x77
	.sleb128 64
	.uleb128 0x1f
	.long	.LLRL15
	.long	0xbe6
	.uleb128 0x5
	.long	.LASF88
	.byte	0x36
	.byte	0x18
	.long	0xba
	.long	.LLST16
	.long	.LVUS16
	.uleb128 0xc
	.long	0xf28
	.quad	.LBI49
	.byte	.LVU40
	.long	.LLRL17
	.byte	0x39
	.byte	0x14
	.long	0x9a0
	.uleb128 0x4
	.long	0xf37
	.long	.LLST18
	.long	.LVUS18
	.byte	0
	.uleb128 0x12
	.long	0xfa0
	.quad	.LBI53
	.byte	.LVU46
	.quad	.LBB53
	.quad	.LBE53-.LBB53
	.byte	0x3a
	.byte	0x14
	.long	0x9df
	.uleb128 0x4
	.long	0xfbb
	.long	.LLST19
	.long	.LVUS19
	.uleb128 0x4
	.long	0xfaf
	.long	.LLST20
	.long	.LVUS20
	.byte	0
	.uleb128 0xc
	.long	0xf78
	.quad	.LBI55
	.byte	.LVU51
	.long	.LLRL21
	.byte	0x3b
	.byte	0x14
	.long	0xa12
	.uleb128 0x4
	.long	0xf93
	.long	.LLST22
	.long	.LVUS22
	.uleb128 0x4
	.long	0xf87
	.long	.LLST22
	.long	.LVUS22
	.byte	0
	.uleb128 0xc
	.long	0xf28
	.quad	.LBI59
	.byte	.LVU61
	.long	.LLRL24
	.byte	0x3e
	.byte	0x14
	.long	0xa38
	.uleb128 0x4
	.long	0xf37
	.long	.LLST25
	.long	.LVUS25
	.byte	0
	.uleb128 0xc
	.long	0xdd8
	.quad	.LBI62
	.byte	.LVU99
	.long	.LLRL26
	.byte	0x4a
	.byte	0x15
	.long	0xa6b
	.uleb128 0x4
	.long	0xdf3
	.long	.LLST27
	.long	.LVUS27
	.uleb128 0x4
	.long	0xde7
	.long	.LLST28
	.long	.LVUS28
	.byte	0
	.uleb128 0x12
	.long	0xfa0
	.quad	.LBI66
	.byte	.LVU67
	.quad	.LBB66
	.quad	.LBE66-.LBB66
	.byte	0x3f
	.byte	0x14
	.long	0xaaa
	.uleb128 0x4
	.long	0xfbb
	.long	.LLST29
	.long	.LVUS29
	.uleb128 0x4
	.long	0xfaf
	.long	.LLST30
	.long	.LVUS30
	.byte	0
	.uleb128 0xc
	.long	0xf78
	.quad	.LBI68
	.byte	.LVU72
	.long	.LLRL31
	.byte	0x40
	.byte	0x14
	.long	0xadd
	.uleb128 0x4
	.long	0xf93
	.long	.LLST32
	.long	.LVUS32
	.uleb128 0x4
	.long	0xf87
	.long	.LLST32
	.long	.LVUS32
	.byte	0
	.uleb128 0xc
	.long	0xff0
	.quad	.LBI71
	.byte	.LVU76
	.long	.LLRL34
	.byte	0x43
	.byte	0x17
	.long	0xb08
	.uleb128 0x2f
	.long	0x100b
	.uleb128 0x4
	.long	0xfff
	.long	.LLST35
	.long	.LVUS35
	.byte	0
	.uleb128 0xc
	.long	0xf44
	.quad	.LBI74
	.byte	.LVU80
	.long	.LLRL36
	.byte	0x47
	.byte	0x16
	.long	0xb48
	.uleb128 0x4
	.long	0xf6b
	.long	.LLST37
	.long	.LVUS37
	.uleb128 0x4
	.long	0xf5f
	.long	.LLST38
	.long	.LVUS38
	.uleb128 0x4
	.long	0xf53
	.long	.LLST10
	.long	.LVUS10
	.byte	0
	.uleb128 0x12
	.long	0xfc8
	.quad	.LBI80
	.byte	.LVU89
	.quad	.LBB80
	.quad	.LBE80-.LBB80
	.byte	0x48
	.byte	0x16
	.long	0xb87
	.uleb128 0x4
	.long	0xfe3
	.long	.LLST40
	.long	.LVUS40
	.uleb128 0x4
	.long	0xfd7
	.long	.LLST41
	.long	.LVUS41
	.byte	0
	.uleb128 0x12
	.long	0xf0c
	.quad	.LBI82
	.byte	.LVU94
	.quad	.LBB82
	.quad	.LBE82-.LBB82
	.byte	0x49
	.byte	0x15
	.long	0xbb9
	.uleb128 0x4
	.long	0xf1b
	.long	.LLST42
	.long	.LVUS42
	.byte	0
	.uleb128 0xa
	.quad	.LVL12
	.long	0x45c
	.long	0xbd1
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x2
	.byte	0x7c
	.sleb128 0
	.byte	0
	.uleb128 0xb
	.quad	.LVL17
	.long	0x45c
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x2
	.byte	0x7c
	.sleb128 0
	.byte	0
	.byte	0
	.uleb128 0x1f
	.long	.LLRL43
	.long	0xc01
	.uleb128 0xf
	.string	"i"
	.byte	0x4f
	.byte	0xe
	.long	0x38
	.long	.LLST44
	.long	.LVUS44
	.byte	0
	.uleb128 0x1c
	.quad	.LVL2
	.long	0x493
	.uleb128 0x1c
	.quad	.LVL4
	.long	0x493
	.uleb128 0xa
	.quad	.LVL7
	.long	0x477
	.long	0xc3f
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x8
	.byte	0x7e
	.sleb128 0
	.byte	0x8
	.byte	0x20
	.byte	0x24
	.byte	0x8
	.byte	0x20
	.byte	0x26
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x51
	.uleb128 0x2
	.byte	0x7c
	.sleb128 0
	.byte	0
	.uleb128 0xa
	.quad	.LVL27
	.long	0x4b7
	.long	0xc57
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x2
	.byte	0x7c
	.sleb128 0
	.byte	0
	.uleb128 0xa
	.quad	.LVL28
	.long	0x4a0
	.long	0xc6f
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x2
	.byte	0x7c
	.sleb128 0
	.byte	0
	.uleb128 0x1c
	.quad	.LVL33
	.long	0x10b0
	.byte	0
	.uleb128 0x15
	.long	0x2d8
	.long	0xc8d
	.uleb128 0x16
	.long	0x4b
	.byte	0x1
	.byte	0
	.uleb128 0x15
	.long	0x38
	.long	0xc9d
	.uleb128 0x16
	.long	0x4b
	.byte	0x7
	.byte	0
	.uleb128 0x1e
	.long	.LASF89
	.byte	0xd
	.byte	0x7
	.long	0x59
	.quad	.LFB5699
	.quad	.LFE5699-.LFB5699
	.uleb128 0x1
	.byte	0x9c
	.long	0xdd8
	.uleb128 0x1b
	.long	.LASF74
	.byte	0xd
	.byte	0x2d
	.long	0x59
	.long	.LLST45
	.long	.LVUS45
	.uleb128 0x24
	.long	.LASF91
	.byte	0xe
	.byte	0x12
	.long	0x52
	.uleb128 0x2
	.byte	0x91
	.sleb128 -60
	.uleb128 0xf
	.string	"hit"
	.byte	0x10
	.byte	0x13
	.long	0xba
	.long	.LLST46
	.long	.LVUS46
	.uleb128 0x5
	.long	.LASF92
	.byte	0x10
	.byte	0x1c
	.long	0xba
	.long	.LLST47
	.long	.LVUS47
	.uleb128 0xf
	.string	"x"
	.byte	0x11
	.byte	0xc
	.long	0x2a
	.long	.LLST48
	.long	.LVUS48
	.uleb128 0xf
	.string	"y"
	.byte	0x11
	.byte	0xf
	.long	0x2a
	.long	.LLST49
	.long	.LVUS49
	.uleb128 0x5
	.long	.LASF93
	.byte	0x11
	.byte	0x12
	.long	0x2a
	.long	.LLST50
	.long	.LVUS50
	.uleb128 0x1f
	.long	.LLRL51
	.long	0xd83
	.uleb128 0x5
	.long	.LASF88
	.byte	0x12
	.byte	0x18
	.long	0xba
	.long	.LLST52
	.long	.LVUS52
	.uleb128 0xa
	.quad	.LVL39
	.long	0x4ce
	.long	0xd6e
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x2
	.byte	0x7c
	.sleb128 0
	.byte	0
	.uleb128 0xb
	.quad	.LVL41
	.long	0x4ce
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x2
	.byte	0x7c
	.sleb128 0
	.byte	0
	.byte	0
	.uleb128 0xa
	.quad	.LVL36
	.long	0x4ea
	.long	0xd9a
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x1
	.byte	0x30
	.byte	0
	.uleb128 0xa
	.quad	.LVL48
	.long	0x4b7
	.long	0xdb2
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x2
	.byte	0x7c
	.sleb128 0
	.byte	0
	.uleb128 0xa
	.quad	.LVL49
	.long	0x4a0
	.long	0xdca
	.uleb128 0x1
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x2
	.byte	0x7c
	.sleb128 0
	.byte	0
	.uleb128 0x1c
	.quad	.LVL53
	.long	0x10b0
	.byte	0
	.uleb128 0x19
	.long	.LASF94
	.byte	0x3
	.byte	0x77
	.long	0x2c0
	.long	0xe00
	.uleb128 0xd
	.string	"__A"
	.byte	0x3
	.byte	0x77
	.byte	0x1b
	.long	0x2c0
	.uleb128 0xd
	.string	"__B"
	.byte	0x3
	.byte	0x77
	.byte	0x28
	.long	0x2c0
	.byte	0
	.uleb128 0x13
	.long	.LASF95
	.value	0x4f1
	.long	0x2c0
	.long	0xe70
	.uleb128 0x3
	.string	"__A"
	.value	0x4f1
	.byte	0x17
	.long	0x38
	.uleb128 0x3
	.string	"__B"
	.value	0x4f1
	.byte	0x20
	.long	0x38
	.uleb128 0x3
	.string	"__C"
	.value	0x4f1
	.byte	0x29
	.long	0x38
	.uleb128 0x3
	.string	"__D"
	.value	0x4f1
	.byte	0x32
	.long	0x38
	.uleb128 0x3
	.string	"__E"
	.value	0x4f2
	.byte	0x9
	.long	0x38
	.uleb128 0x3
	.string	"__F"
	.value	0x4f2
	.byte	0x12
	.long	0x38
	.uleb128 0x3
	.string	"__G"
	.value	0x4f2
	.byte	0x1b
	.long	0x38
	.uleb128 0x3
	.string	"__H"
	.value	0x4f2
	.byte	0x24
	.long	0x38
	.byte	0
	.uleb128 0x13
	.long	.LASF96
	.value	0x4e8
	.long	0x2a8
	.long	0xee0
	.uleb128 0x3
	.string	"__A"
	.value	0x4e8
	.byte	0x16
	.long	0x259
	.uleb128 0x3
	.string	"__B"
	.value	0x4e8
	.byte	0x21
	.long	0x259
	.uleb128 0x3
	.string	"__C"
	.value	0x4e8
	.byte	0x2c
	.long	0x259
	.uleb128 0x3
	.string	"__D"
	.value	0x4e8
	.byte	0x37
	.long	0x259
	.uleb128 0x3
	.string	"__E"
	.value	0x4e9
	.byte	0xf
	.long	0x259
	.uleb128 0x3
	.string	"__F"
	.value	0x4e9
	.byte	0x1a
	.long	0x259
	.uleb128 0x3
	.string	"__G"
	.value	0x4e9
	.byte	0x25
	.long	0x259
	.uleb128 0x3
	.string	"__H"
	.value	0x4e9
	.byte	0x30
	.long	0x259
	.byte	0
	.uleb128 0x30
	.long	.LASF97
	.byte	0x2
	.value	0x399
	.byte	0x1
	.byte	0x3
	.long	0xf07
	.uleb128 0x3
	.string	"__P"
	.value	0x399
	.byte	0x1e
	.long	0xf07
	.uleb128 0x3
	.string	"__A"
	.value	0x399
	.byte	0x2b
	.long	0x2c0
	.byte	0
	.uleb128 0x7
	.long	0x2c0
	.uleb128 0x13
	.long	.LASF98
	.value	0x1d9
	.long	0x2c0
	.long	0xf28
	.uleb128 0x3
	.string	"__A"
	.value	0x1d9
	.byte	0x1c
	.long	0x2a8
	.byte	0
	.uleb128 0x13
	.long	.LASF99
	.value	0x1cd
	.long	0x2a8
	.long	0xf44
	.uleb128 0x3
	.string	"__A"
	.value	0x1cd
	.byte	0x1d
	.long	0x2c0
	.byte	0
	.uleb128 0x13
	.long	.LASF100
	.value	0x194
	.long	0x2a8
	.long	0xf78
	.uleb128 0x3
	.string	"__X"
	.value	0x194
	.byte	0x17
	.long	0x2a8
	.uleb128 0x3
	.string	"__Y"
	.value	0x194
	.byte	0x23
	.long	0x2a8
	.uleb128 0x3
	.string	"__P"
	.value	0x194
	.byte	0x32
	.long	0x7c
	.byte	0
	.uleb128 0x13
	.long	.LASF101
	.value	0x13e
	.long	0x2a8
	.long	0xfa0
	.uleb128 0x3
	.string	"__A"
	.value	0x13e
	.byte	0x17
	.long	0x2a8
	.uleb128 0x3
	.string	"__B"
	.value	0x13e
	.byte	0x23
	.long	0x2a8
	.byte	0
	.uleb128 0x19
	.long	.LASF102
	.byte	0x2
	.byte	0xf1
	.long	0x2a8
	.long	0xfc8
	.uleb128 0xd
	.string	"__A"
	.byte	0x2
	.byte	0xf1
	.byte	0x17
	.long	0x2a8
	.uleb128 0xd
	.string	"__B"
	.byte	0x2
	.byte	0xf1
	.byte	0x23
	.long	0x2a8
	.byte	0
	.uleb128 0x19
	.long	.LASF103
	.byte	0x2
	.byte	0xac
	.long	0x2a8
	.long	0xff0
	.uleb128 0xd
	.string	"__A"
	.byte	0x2
	.byte	0xac
	.byte	0x17
	.long	0x2a8
	.uleb128 0xd
	.string	"__B"
	.byte	0x2
	.byte	0xac
	.byte	0x23
	.long	0x2a8
	.byte	0
	.uleb128 0x19
	.long	.LASF104
	.byte	0x2
	.byte	0x93
	.long	0x2a8
	.long	0x1018
	.uleb128 0xd
	.string	"__A"
	.byte	0x2
	.byte	0x93
	.byte	0x17
	.long	0x2a8
	.uleb128 0xd
	.string	"__B"
	.byte	0x2
	.byte	0x93
	.byte	0x23
	.long	0x2a8
	.byte	0
	.uleb128 0x31
	.long	.LASF105
	.byte	0x5
	.value	0x16a
	.byte	0x1
	.long	0x38
	.byte	0x3
	.long	0x1038
	.uleb128 0x32
	.long	.LASF106
	.byte	0x5
	.value	0x16a
	.byte	0x1
	.long	0x248
	.byte	0
	.uleb128 0x19
	.long	.LASF107
	.byte	0x4
	.byte	0x6e
	.long	0x38
	.long	0x1055
	.uleb128 0x33
	.long	.LASF108
	.byte	0x4
	.byte	0x6e
	.byte	0x20
	.long	0x24d
	.uleb128 0x22
	.byte	0
	.uleb128 0x25
	.uleb128 0x2e
	.byte	0x9e
	.uleb128 0x2c
	.byte	0x55
	.byte	0x73
	.byte	0x61
	.byte	0x67
	.byte	0x65
	.byte	0x20
	.byte	0x6f
	.byte	0x66
	.byte	0x20
	.byte	0x63
	.byte	0x6f
	.byte	0x6d
	.byte	0x6d
	.byte	0x61
	.byte	0x6e
	.byte	0x64
	.byte	0x3a
	.byte	0x20
	.byte	0x2e
	.byte	0x2f
	.byte	0x70
	.byte	0x69
	.byte	0x2e
	.byte	0x6f
	.byte	0x75
	.byte	0x74
	.byte	0x20
	.byte	0x23
	.byte	0x54
	.byte	0x68
	.byte	0x72
	.byte	0x65
	.byte	0x61
	.byte	0x64
	.byte	0x20
	.byte	0x23
	.byte	0x54
	.byte	0x6f
	.byte	0x73
	.byte	0x73
	.byte	0x65
	.byte	0x73
	.byte	0xa
	.byte	0
	.uleb128 0x34
	.long	.LASF112
	.long	.LASF113
	.byte	0x10
	.byte	0
	.uleb128 0x25
	.uleb128 0x1e
	.byte	0x9e
	.uleb128 0x1c
	.byte	0x65
	.byte	0x2e
	.byte	0x67
	.byte	0x2e
	.byte	0x20
	.byte	0x2e
	.byte	0x2f
	.byte	0x70
	.byte	0x69
	.byte	0x2e
	.byte	0x6f
	.byte	0x75
	.byte	0x74
	.byte	0x20
	.byte	0x38
	.byte	0x20
	.byte	0x31
	.byte	0x30
	.byte	0x30
	.byte	0x30
	.byte	0x30
	.byte	0x30
	.byte	0x30
	.byte	0x30
	.byte	0x30
	.byte	0x30
	.byte	0xa
	.byte	0
	.uleb128 0x35
	.long	.LASF114
	.long	.LASF114
	.byte	0
	.section	.debug_abbrev,"",@progbits
.Ldebug_abbrev0:
	.uleb128 0x1
	.uleb128 0x49
	.byte	0
	.uleb128 0x2
	.uleb128 0x18
	.uleb128 0x7e
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x2
	.uleb128 0x5
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x3
	.uleb128 0x5
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 2
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x4
	.uleb128 0x5
	.byte	0
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x17
	.uleb128 0x2137
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x5
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x17
	.uleb128 0x2137
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x6
	.uleb128 0x16
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x7
	.uleb128 0xf
	.byte	0
	.uleb128 0xb
	.uleb128 0x21
	.sleb128 8
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x8
	.uleb128 0x24
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3e
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0xe
	.byte	0
	.byte	0
	.uleb128 0x9
	.uleb128 0xd
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x38
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0xa
	.uleb128 0x48
	.byte	0x1
	.uleb128 0x7d
	.uleb128 0x1
	.uleb128 0x7f
	.uleb128 0x13
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0xb
	.uleb128 0x48
	.byte	0x1
	.uleb128 0x7d
	.uleb128 0x1
	.uleb128 0x7f
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0xc
	.uleb128 0x1d
	.byte	0x1
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x52
	.uleb128 0x1
	.uleb128 0x2138
	.uleb128 0xb
	.uleb128 0x55
	.uleb128 0x17
	.uleb128 0x58
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x59
	.uleb128 0xb
	.uleb128 0x57
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0xd
	.uleb128 0x5
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0xe
	.uleb128 0xd
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 11
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0xf
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0x8
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x17
	.uleb128 0x2137
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x10
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x3c
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x11
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x3c
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x12
	.uleb128 0x1d
	.byte	0x1
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x52
	.uleb128 0x1
	.uleb128 0x2138
	.uleb128 0xb
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x58
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x59
	.uleb128 0xb
	.uleb128 0x57
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x13
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 2
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x39
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x20
	.uleb128 0x21
	.sleb128 3
	.uleb128 0x34
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x14
	.uleb128 0x37
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x15
	.uleb128 0x1
	.byte	0x1
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x16
	.uleb128 0x21
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2f
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x17
	.uleb128 0x1
	.byte	0x1
	.uleb128 0x2107
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x18
	.uleb128 0x21
	.byte	0
	.uleb128 0x2f
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x19
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x20
	.uleb128 0x21
	.sleb128 3
	.uleb128 0x34
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x1a
	.uleb128 0x26
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x1b
	.uleb128 0x5
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x17
	.uleb128 0x2137
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x1c
	.uleb128 0x48
	.byte	0
	.uleb128 0x7d
	.uleb128 0x1
	.uleb128 0x7f
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x1d
	.uleb128 0x13
	.byte	0x1
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x1e
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x40
	.uleb128 0x18
	.uleb128 0x7a
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x1f
	.uleb128 0xb
	.byte	0x1
	.uleb128 0x55
	.uleb128 0x17
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x20
	.uleb128 0x17
	.byte	0x1
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 11
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0x21
	.sleb128 9
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x21
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x22
	.uleb128 0x18
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x23
	.uleb128 0xb
	.byte	0x1
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x24
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 1
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x25
	.uleb128 0x36
	.byte	0
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x26
	.uleb128 0x11
	.byte	0x1
	.uleb128 0x25
	.uleb128 0xe
	.uleb128 0x13
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0x1f
	.uleb128 0x1b
	.uleb128 0x1f
	.uleb128 0x55
	.uleb128 0x17
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x10
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x27
	.uleb128 0x24
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3e
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0x8
	.byte	0
	.byte	0
	.uleb128 0x28
	.uleb128 0xf
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x29
	.uleb128 0x17
	.byte	0x1
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x2a
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x3c
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x2b
	.uleb128 0x15
	.byte	0x1
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x2c
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x3c
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x2d
	.uleb128 0x2e
	.byte	0
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x3c
	.uleb128 0x19
	.byte	0
	.byte	0
	.uleb128 0x2e
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x2f
	.uleb128 0x5
	.byte	0
	.uleb128 0x31
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x30
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x20
	.uleb128 0xb
	.uleb128 0x34
	.uleb128 0x19
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x31
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x27
	.uleb128 0x19
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x20
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x32
	.uleb128 0x5
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0x5
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x33
	.uleb128 0x5
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x34
	.uleb128 0x2e
	.byte	0
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3c
	.uleb128 0x19
	.uleb128 0x6e
	.uleb128 0xe
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x35
	.uleb128 0x2e
	.byte	0
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x3c
	.uleb128 0x19
	.uleb128 0x6e
	.uleb128 0xe
	.uleb128 0x3
	.uleb128 0xe
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_loclists,"",@progbits
	.long	.Ldebug_loc3-.Ldebug_loc2
.Ldebug_loc2:
	.value	0x5
	.byte	0x8
	.byte	0
	.long	0
.Ldebug_loc0:
.LVUS53:
	.uleb128 0
	.uleb128 .LVU186
	.uleb128 .LVU186
	.uleb128 .LVU196
	.uleb128 .LVU196
	.uleb128 .LVU197
	.uleb128 .LVU197
	.uleb128 0
.LLST53:
	.byte	0x6
	.quad	.LVL54
	.byte	0x4
	.uleb128 .LVL54-.LVL54
	.uleb128 .LVL56-.LVL54
	.uleb128 0x1
	.byte	0x55
	.byte	0x4
	.uleb128 .LVL56-.LVL54
	.uleb128 .LVL59-.LVL54
	.uleb128 0x4
	.byte	0xa3
	.uleb128 0x1
	.byte	0x55
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL59-.LVL54
	.uleb128 .LVL60-.LVL54
	.uleb128 0x1
	.byte	0x55
	.byte	0x4
	.uleb128 .LVL60-.LVL54
	.uleb128 .LFE5701-.LVL54
	.uleb128 0x4
	.byte	0xa3
	.uleb128 0x1
	.byte	0x55
	.byte	0x9f
	.byte	0
.LVUS54:
	.uleb128 0
	.uleb128 .LVU187
	.uleb128 .LVU187
	.uleb128 .LVU196
	.uleb128 .LVU196
	.uleb128 .LVU202
	.uleb128 .LVU202
	.uleb128 .LVU228
	.uleb128 .LVU228
	.uleb128 0
.LLST54:
	.byte	0x6
	.quad	.LVL54
	.byte	0x4
	.uleb128 .LVL54-.LVL54
	.uleb128 .LVL57-1-.LVL54
	.uleb128 0x1
	.byte	0x54
	.byte	0x4
	.uleb128 .LVL57-1-.LVL54
	.uleb128 .LVL59-.LVL54
	.uleb128 0x4
	.byte	0xa3
	.uleb128 0x1
	.byte	0x54
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL59-.LVL54
	.uleb128 .LVL62-.LVL54
	.uleb128 0x1
	.byte	0x54
	.byte	0x4
	.uleb128 .LVL62-.LVL54
	.uleb128 .LVL73-.LVL54
	.uleb128 0x1
	.byte	0x53
	.byte	0x4
	.uleb128 .LVL73-.LVL54
	.uleb128 .LFE5701-.LVL54
	.uleb128 0x4
	.byte	0xa3
	.uleb128 0x1
	.byte	0x54
	.byte	0x9f
	.byte	0
.LVUS55:
	.uleb128 .LVU204
	.uleb128 .LVU230
.LLST55:
	.byte	0x8
	.quad	.LVL64
	.uleb128 .LVL74-.LVL64
	.uleb128 0x1
	.byte	0x56
	.byte	0
.LVUS56:
	.uleb128 .LVU212
	.uleb128 .LVU215
	.uleb128 .LVU215
	.uleb128 .LVU216
	.uleb128 .LVU216
	.uleb128 0
.LLST56:
	.byte	0x6
	.quad	.LVL66
	.byte	0x4
	.uleb128 .LVL66-.LVL66
	.uleb128 .LVL67-.LVL66
	.uleb128 0x9
	.byte	0x70
	.sleb128 0
	.byte	0x8
	.byte	0x20
	.byte	0x24
	.byte	0x8
	.byte	0x20
	.byte	0x26
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL67-.LVL66
	.uleb128 .LVL68-.LVL66
	.uleb128 0x1
	.byte	0x50
	.byte	0x4
	.uleb128 .LVL68-.LVL66
	.uleb128 .LFE5701-.LVL66
	.uleb128 0xc
	.byte	0x91
	.sleb128 -72
	.byte	0x94
	.byte	0x4
	.byte	0x8
	.byte	0x20
	.byte	0x24
	.byte	0x8
	.byte	0x20
	.byte	0x26
	.byte	0x9f
	.byte	0
.LVUS57:
	.uleb128 .LVU218
	.uleb128 .LVU221
	.uleb128 .LVU221
	.uleb128 0
.LLST57:
	.byte	0x6
	.quad	.LVL69
	.byte	0x4
	.uleb128 .LVL69-.LVL69
	.uleb128 .LVL70-1-.LVL69
	.uleb128 0x1
	.byte	0x50
	.byte	0x4
	.uleb128 .LVL70-1-.LVL69
	.uleb128 .LFE5701-.LVL69
	.uleb128 0x1
	.byte	0x5c
	.byte	0
.LVUS58:
	.uleb128 .LVU242
	.uleb128 .LVU253
.LLST58:
	.byte	0x8
	.quad	.LVL80
	.uleb128 .LVL81-1-.LVL80
	.uleb128 0x1b
	.byte	0x3
	.quad	total_hit
	.byte	0x6
	.byte	0x32
	.byte	0x24
	.byte	0xa8
	.uleb128 0x31
	.byte	0xa8
	.uleb128 0x2a
	.byte	0x91
	.sleb128 -72
	.byte	0x94
	.byte	0x4
	.byte	0xa8
	.uleb128 0x38
	.byte	0xa8
	.uleb128 0x2a
	.byte	0x1b
	.byte	0x9f
	.byte	0
.LVUS59:
	.uleb128 .LVU223
	.uleb128 .LVU224
	.uleb128 .LVU224
	.uleb128 .LVU230
	.uleb128 .LVU230
	.uleb128 .LVU234
	.uleb128 .LVU234
	.uleb128 0
.LLST59:
	.byte	0x6
	.quad	.LVL71
	.byte	0x4
	.uleb128 .LVL71-.LVL71
	.uleb128 .LVL72-1-.LVL71
	.uleb128 0x1
	.byte	0x50
	.byte	0x4
	.uleb128 .LVL72-1-.LVL71
	.uleb128 .LVL74-.LVL71
	.uleb128 0x1
	.byte	0x5e
	.byte	0x4
	.uleb128 .LVL74-.LVL71
	.uleb128 .LVL77-.LVL71
	.uleb128 0x1
	.byte	0x53
	.byte	0x4
	.uleb128 .LVL77-.LVL71
	.uleb128 .LFE5701-.LVL71
	.uleb128 0x1
	.byte	0x5e
	.byte	0
.LVUS66:
	.uleb128 .LVU226
	.uleb128 .LVU230
	.uleb128 .LVU230
	.uleb128 .LVU232
	.uleb128 .LVU232
	.uleb128 .LVU233
	.uleb128 .LVU233
	.uleb128 .LVU234
	.uleb128 .LVU234
	.uleb128 .LVU240
.LLST66:
	.byte	0x6
	.quad	.LVL72
	.byte	0x4
	.uleb128 .LVL72-.LVL72
	.uleb128 .LVL74-.LVL72
	.uleb128 0x2
	.byte	0x30
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL74-.LVL72
	.uleb128 .LVL75-.LVL72
	.uleb128 0x8
	.byte	0x7f
	.sleb128 0
	.byte	0x73
	.sleb128 0
	.byte	0x1c
	.byte	0x33
	.byte	0x25
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL75-.LVL72
	.uleb128 .LVL76-.LVL72
	.uleb128 0xa
	.byte	0x7f
	.sleb128 0
	.byte	0x73
	.sleb128 0
	.byte	0x1c
	.byte	0x33
	.byte	0x25
	.byte	0x23
	.uleb128 0x1
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL76-.LVL72
	.uleb128 .LVL77-.LVL72
	.uleb128 0xc
	.byte	0x7f
	.sleb128 0
	.byte	0x73
	.sleb128 0
	.byte	0x1c
	.byte	0x38
	.byte	0x1c
	.byte	0x33
	.byte	0x25
	.byte	0x23
	.uleb128 0x1
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL77-.LVL72
	.uleb128 .LVL80-.LVL72
	.uleb128 0xc
	.byte	0x7f
	.sleb128 0
	.byte	0x7e
	.sleb128 0
	.byte	0x1c
	.byte	0x38
	.byte	0x1c
	.byte	0x33
	.byte	0x25
	.byte	0x23
	.uleb128 0x1
	.byte	0x9f
	.byte	0
.LVUS67:
	.uleb128 .LVU234
	.uleb128 .LVU237
	.uleb128 .LVU237
	.uleb128 .LVU239
.LLST67:
	.byte	0x6
	.quad	.LVL77
	.byte	0x4
	.uleb128 .LVL77-.LVL77
	.uleb128 .LVL78-.LVL77
	.uleb128 0x8
	.byte	0x73
	.sleb128 0
	.byte	0x7e
	.sleb128 0
	.byte	0x1c
	.byte	0x33
	.byte	0x25
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL78-.LVL77
	.uleb128 .LVL79-.LVL77
	.uleb128 0xa
	.byte	0x73
	.sleb128 0
	.byte	0x7e
	.sleb128 0
	.byte	0x1c
	.byte	0x38
	.byte	0x1c
	.byte	0x33
	.byte	0x25
	.byte	0x9f
	.byte	0
.LVUS60:
	.uleb128 .LVU183
	.uleb128 .LVU187
.LLST60:
	.byte	0x8
	.quad	.LVL55
	.uleb128 .LVL57-.LVL55
	.uleb128 0x6
	.byte	0xa0
	.long	.Ldebug_info0+4181
	.sleb128 0
	.byte	0
.LVUS61:
	.uleb128 .LVU189
	.uleb128 .LVU192
.LLST61:
	.byte	0x8
	.quad	.LVL57
	.uleb128 .LVL58-.LVL57
	.uleb128 0x6
	.byte	0xa0
	.long	.Ldebug_info0+4240
	.sleb128 0
	.byte	0
.LVUS63:
	.uleb128 .LVU199
	.uleb128 .LVU203
.LLST63:
	.byte	0x8
	.quad	.LVL61
	.uleb128 .LVL63-1-.LVL61
	.uleb128 0x1
	.byte	0x55
	.byte	0
.LVUS65:
	.uleb128 .LVU206
	.uleb128 .LVU209
.LLST65:
	.byte	0x8
	.quad	.LVL64
	.uleb128 .LVL65-1-.LVL64
	.uleb128 0x1
	.byte	0x55
	.byte	0
.LVUS69:
	.uleb128 .LVU243
	.uleb128 .LVU253
.LLST69:
	.byte	0x8
	.quad	.LVL80
	.uleb128 .LVL81-.LVL80
	.uleb128 0xa
	.byte	0x3
	.quad	.LC6
	.byte	0x9f
	.byte	0
.LVUS0:
	.uleb128 0
	.uleb128 .LVU11
	.uleb128 .LVU11
	.uleb128 .LVU33
	.uleb128 .LVU33
	.uleb128 .LVU128
	.uleb128 .LVU128
	.uleb128 .LVU129
	.uleb128 .LVU129
	.uleb128 0
.LLST0:
	.byte	0x6
	.quad	.LVL0
	.byte	0x4
	.uleb128 .LVL0-.LVL0
	.uleb128 .LVL2-1-.LVL0
	.uleb128 0x1
	.byte	0x55
	.byte	0x4
	.uleb128 .LVL2-1-.LVL0
	.uleb128 .LVL8-.LVL0
	.uleb128 0x1
	.byte	0x53
	.byte	0x4
	.uleb128 .LVL8-.LVL0
	.uleb128 .LVL31-.LVL0
	.uleb128 0x4
	.byte	0xa3
	.uleb128 0x1
	.byte	0x55
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL31-.LVL0
	.uleb128 .LVL32-.LVL0
	.uleb128 0x1
	.byte	0x53
	.byte	0x4
	.uleb128 .LVL32-.LVL0
	.uleb128 .LFE5700-.LVL0
	.uleb128 0x4
	.byte	0xa3
	.uleb128 0x1
	.byte	0x55
	.byte	0x9f
	.byte	0
.LVUS1:
	.uleb128 .LVU22
	.uleb128 .LVU24
	.uleb128 .LVU24
	.uleb128 .LVU24
	.uleb128 .LVU24
	.uleb128 .LVU26
	.uleb128 .LVU26
	.uleb128 .LVU28
	.uleb128 .LVU28
	.uleb128 .LVU29
	.uleb128 .LVU29
	.uleb128 .LVU127
	.uleb128 .LVU128
	.uleb128 0
.LLST1:
	.byte	0x6
	.quad	.LVL3
	.byte	0x4
	.uleb128 .LVL3-.LVL3
	.uleb128 .LVL4-1-.LVL3
	.uleb128 0xd
	.byte	0x70
	.sleb128 0
	.byte	0x8
	.byte	0x20
	.byte	0x24
	.byte	0x8
	.byte	0x20
	.byte	0x26
	.byte	0x9f
	.byte	0x93
	.uleb128 0x8
	.byte	0x93
	.uleb128 0x8
	.byte	0x4
	.uleb128 .LVL4-1-.LVL3
	.uleb128 .LVL4-.LVL3
	.uleb128 0xd
	.byte	0x7e
	.sleb128 0
	.byte	0x8
	.byte	0x20
	.byte	0x24
	.byte	0x8
	.byte	0x20
	.byte	0x26
	.byte	0x9f
	.byte	0x93
	.uleb128 0x8
	.byte	0x93
	.uleb128 0x8
	.byte	0x4
	.uleb128 .LVL4-.LVL3
	.uleb128 .LVL5-.LVL3
	.uleb128 0x16
	.byte	0x7e
	.sleb128 0
	.byte	0x8
	.byte	0x20
	.byte	0x24
	.byte	0x8
	.byte	0x20
	.byte	0x26
	.byte	0x9f
	.byte	0x93
	.uleb128 0x8
	.byte	0x70
	.sleb128 0
	.byte	0x8
	.byte	0x20
	.byte	0x24
	.byte	0x8
	.byte	0x20
	.byte	0x26
	.byte	0x9f
	.byte	0x93
	.uleb128 0x8
	.byte	0x4
	.uleb128 .LVL5-.LVL3
	.uleb128 .LVL6-.LVL3
	.uleb128 0xe
	.byte	0x55
	.byte	0x93
	.uleb128 0x8
	.byte	0x70
	.sleb128 0
	.byte	0x8
	.byte	0x20
	.byte	0x24
	.byte	0x8
	.byte	0x20
	.byte	0x26
	.byte	0x9f
	.byte	0x93
	.uleb128 0x8
	.byte	0x4
	.uleb128 .LVL6-.LVL3
	.uleb128 .LVL7-1-.LVL3
	.uleb128 0x6
	.byte	0x55
	.byte	0x93
	.uleb128 0x8
	.byte	0x54
	.byte	0x93
	.uleb128 0x8
	.byte	0x4
	.uleb128 .LVL7-1-.LVL3
	.uleb128 .LVL30-.LVL3
	.uleb128 0xd
	.byte	0x7e
	.sleb128 0
	.byte	0x8
	.byte	0x20
	.byte	0x24
	.byte	0x8
	.byte	0x20
	.byte	0x26
	.byte	0x9f
	.byte	0x93
	.uleb128 0x8
	.byte	0x93
	.uleb128 0x8
	.byte	0x4
	.uleb128 .LVL31-.LVL3
	.uleb128 .LFE5700-.LVL3
	.uleb128 0xd
	.byte	0x7e
	.sleb128 0
	.byte	0x8
	.byte	0x20
	.byte	0x24
	.byte	0x8
	.byte	0x20
	.byte	0x26
	.byte	0x9f
	.byte	0x93
	.uleb128 0x8
	.byte	0x93
	.uleb128 0x8
	.byte	0
.LVUS2:
	.uleb128 .LVU7
	.uleb128 .LVU110
	.uleb128 .LVU128
	.uleb128 .LVU129
.LLST2:
	.byte	0x6
	.quad	.LVL1
	.byte	0x4
	.uleb128 .LVL1-.LVL1
	.uleb128 .LVL26-.LVL1
	.uleb128 0x2
	.byte	0x30
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL31-.LVL1
	.uleb128 .LVL32-.LVL1
	.uleb128 0x2
	.byte	0x30
	.byte	0x9f
	.byte	0
.LVUS3:
	.uleb128 .LVU13
	.uleb128 .LVU126
	.uleb128 .LVU126
	.uleb128 .LVU128
	.uleb128 .LVU128
	.uleb128 0
.LLST3:
	.byte	0x6
	.quad	.LVL3
	.byte	0x4
	.uleb128 .LVL3-.LVL3
	.uleb128 .LVL29-.LVL3
	.uleb128 0x1
	.byte	0x5d
	.byte	0x4
	.uleb128 .LVL29-.LVL3
	.uleb128 .LVL31-.LVL3
	.uleb128 0x15
	.byte	0xa3
	.uleb128 0x1
	.byte	0x55
	.byte	0x23
	.uleb128 0x7
	.byte	0xa3
	.uleb128 0x1
	.byte	0x55
	.byte	0xa3
	.uleb128 0x1
	.byte	0x55
	.byte	0x30
	.byte	0x2d
	.byte	0x28
	.value	0x1
	.byte	0x16
	.byte	0x13
	.byte	0x33
	.byte	0x26
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL31-.LVL3
	.uleb128 .LFE5700-.LVL3
	.uleb128 0x1
	.byte	0x5d
	.byte	0
.LVUS4:
	.uleb128 .LVU39
	.uleb128 .LVU44
.LLST4:
	.byte	0x8
	.quad	.LVL12
	.uleb128 .LVL13-.LVL12
	.uleb128 0x1
	.byte	0x61
	.byte	0
.LVUS5:
	.uleb128 .LVU60
	.uleb128 .LVU65
.LLST5:
	.byte	0x8
	.quad	.LVL17
	.uleb128 .LVL18-.LVL17
	.uleb128 0x1
	.byte	0x61
	.byte	0
.LVUS6:
	.uleb128 .LVU97
	.uleb128 .LVU105
.LLST6:
	.byte	0x8
	.quad	.LVL24
	.uleb128 .LVL26-.LVL24
	.uleb128 0x1
	.byte	0x61
	.byte	0
.LVUS7:
	.uleb128 .LVU36
	.uleb128 .LVU38
	.uleb128 .LVU38
	.uleb128 .LVU102
.LLST7:
	.byte	0x6
	.quad	.LVL10
	.byte	0x4
	.uleb128 .LVL10-.LVL10
	.uleb128 .LVL11-.LVL10
	.uleb128 0x1
	.byte	0x62
	.byte	0x4
	.uleb128 .LVL11-.LVL10
	.uleb128 .LVL25-.LVL10
	.uleb128 0x2
	.byte	0x77
	.sleb128 0
	.byte	0
.LVUS8:
	.uleb128 .LVU44
	.uleb128 .LVU53
	.uleb128 .LVU53
	.uleb128 .LVU56
.LLST8:
	.byte	0x6
	.quad	.LVL13
	.byte	0x4
	.uleb128 .LVL13-.LVL13
	.uleb128 .LVL14-.LVL13
	.uleb128 0x1
	.byte	0x61
	.byte	0x4
	.uleb128 .LVL14-.LVL13
	.uleb128 .LVL15-.LVL13
	.uleb128 0x2
	.byte	0x77
	.sleb128 32
	.byte	0
.LVUS9:
	.uleb128 .LVU65
	.uleb128 .LVU74
	.uleb128 .LVU83
	.uleb128 .LVU85
.LLST9:
	.byte	0x6
	.quad	.LVL18
	.byte	0x4
	.uleb128 .LVL18-.LVL18
	.uleb128 .LVL19-.LVL18
	.uleb128 0x1
	.byte	0x61
	.byte	0x4
	.uleb128 .LVL20-.LVL18
	.uleb128 .LVL21-.LVL18
	.uleb128 0x1
	.byte	0x61
	.byte	0
.LVUS10:
	.uleb128 .LVU85
	.uleb128 .LVU87
.LLST10:
	.byte	0x8
	.quad	.LVL21
	.uleb128 .LVL22-.LVL21
	.uleb128 0x1
	.byte	0x61
	.byte	0
.LVUS11:
	.uleb128 .LVU19
	.uleb128 0
.LLST11:
	.byte	0x8
	.quad	.LVL3
	.uleb128 .LFE5700-.LVL3
	.uleb128 0x40
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0
.LVUS12:
	.uleb128 .LVU20
	.uleb128 0
.LLST12:
	.byte	0x8
	.quad	.LVL3
	.uleb128 .LFE5700-.LVL3
	.uleb128 0x40
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0
.LVUS13:
	.uleb128 .LVU87
	.uleb128 .LVU97
.LLST13:
	.byte	0x8
	.quad	.LVL22
	.uleb128 .LVL24-.LVL22
	.uleb128 0x1
	.byte	0x61
	.byte	0
.LVUS14:
	.uleb128 .LVU17
	.uleb128 0
.LLST14:
	.byte	0x8
	.quad	.LVL3
	.uleb128 .LFE5700-.LVL3
	.uleb128 0x40
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0
.LVUS16:
	.uleb128 .LVU31
	.uleb128 .LVU35
	.uleb128 .LVU36
	.uleb128 .LVU59
	.uleb128 .LVU59
	.uleb128 .LVU104
	.uleb128 .LVU104
	.uleb128 .LVU105
	.uleb128 .LVU128
	.uleb128 .LVU129
.LLST16:
	.byte	0x6
	.quad	.LVL7
	.byte	0x4
	.uleb128 .LVL7-.LVL7
	.uleb128 .LVL9-.LVL7
	.uleb128 0x2
	.byte	0x30
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL10-.LVL7
	.uleb128 .LVL16-.LVL7
	.uleb128 0x1
	.byte	0x53
	.byte	0x4
	.uleb128 .LVL16-.LVL7
	.uleb128 .LVL25-.LVL7
	.uleb128 0x3
	.byte	0x73
	.sleb128 -1
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL25-.LVL7
	.uleb128 .LVL26-.LVL7
	.uleb128 0x1
	.byte	0x53
	.byte	0x4
	.uleb128 .LVL31-.LVL7
	.uleb128 .LVL32-.LVL7
	.uleb128 0x2
	.byte	0x30
	.byte	0x9f
	.byte	0
.LVUS18:
	.uleb128 .LVU40
	.uleb128 .LVU44
.LLST18:
	.byte	0x8
	.quad	.LVL12
	.uleb128 .LVL13-.LVL12
	.uleb128 0x1
	.byte	0x61
	.byte	0
.LVUS19:
	.uleb128 .LVU46
	.uleb128 .LVU49
.LLST19:
	.byte	0x8
	.quad	.LVL13
	.uleb128 .LVL14-.LVL13
	.uleb128 0x40
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0
.LVUS20:
	.uleb128 .LVU46
	.uleb128 .LVU49
.LLST20:
	.byte	0x8
	.quad	.LVL13
	.uleb128 .LVL14-.LVL13
	.uleb128 0x1
	.byte	0x61
	.byte	0
.LVUS22:
	.uleb128 .LVU51
	.uleb128 .LVU53
.LLST22:
	.byte	0x8
	.quad	.LVL14
	.uleb128 .LVL14-.LVL14
	.uleb128 0x1
	.byte	0x61
	.byte	0
.LVUS25:
	.uleb128 .LVU61
	.uleb128 .LVU65
.LLST25:
	.byte	0x8
	.quad	.LVL17
	.uleb128 .LVL18-.LVL17
	.uleb128 0x1
	.byte	0x61
	.byte	0
.LVUS27:
	.uleb128 .LVU99
	.uleb128 .LVU102
.LLST27:
	.byte	0x8
	.quad	.LVL24
	.uleb128 .LVL25-.LVL24
	.uleb128 0x1
	.byte	0x61
	.byte	0
.LVUS28:
	.uleb128 .LVU99
	.uleb128 .LVU102
.LLST28:
	.byte	0x8
	.quad	.LVL24
	.uleb128 .LVL25-.LVL24
	.uleb128 0x1
	.byte	0x62
	.byte	0
.LVUS29:
	.uleb128 .LVU67
	.uleb128 .LVU70
.LLST29:
	.byte	0x8
	.quad	.LVL18
	.uleb128 .LVL19-.LVL18
	.uleb128 0x40
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x4f000000
	.byte	0x93
	.uleb128 0x4
	.byte	0
.LVUS30:
	.uleb128 .LVU67
	.uleb128 .LVU70
.LLST30:
	.byte	0x8
	.quad	.LVL18
	.uleb128 .LVL19-.LVL18
	.uleb128 0x1
	.byte	0x61
	.byte	0
.LVUS32:
	.uleb128 .LVU72
	.uleb128 .LVU74
.LLST32:
	.byte	0x8
	.quad	.LVL19
	.uleb128 .LVL19-.LVL19
	.uleb128 0x1
	.byte	0x61
	.byte	0
.LVUS35:
	.uleb128 .LVU76
	.uleb128 .LVU78
.LLST35:
	.byte	0x8
	.quad	.LVL19
	.uleb128 .LVL19-.LVL19
	.uleb128 0x2
	.byte	0x77
	.sleb128 32
	.byte	0
.LVUS37:
	.uleb128 .LVU80
	.uleb128 .LVU87
.LLST37:
	.byte	0x8
	.quad	.LVL19
	.uleb128 .LVL22-.LVL19
	.uleb128 0x2
	.byte	0x32
	.byte	0x9f
	.byte	0
.LVUS38:
	.uleb128 .LVU80
	.uleb128 .LVU87
.LLST38:
	.byte	0x8
	.quad	.LVL19
	.uleb128 .LVL22-.LVL19
	.uleb128 0x40
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0
.LVUS40:
	.uleb128 .LVU89
	.uleb128 .LVU92
.LLST40:
	.byte	0x8
	.quad	.LVL22
	.uleb128 .LVL23-.LVL22
	.uleb128 0x40
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0x9e
	.uleb128 0x4
	.long	0x3f800000
	.byte	0x93
	.uleb128 0x4
	.byte	0
.LVUS41:
	.uleb128 .LVU89
	.uleb128 .LVU92
.LLST41:
	.byte	0x8
	.quad	.LVL22
	.uleb128 .LVL23-.LVL22
	.uleb128 0x1
	.byte	0x61
	.byte	0
.LVUS42:
	.uleb128 .LVU94
	.uleb128 .LVU97
.LLST42:
	.byte	0x8
	.quad	.LVL23
	.uleb128 .LVL24-.LVL23
	.uleb128 0x1
	.byte	0x61
	.byte	0
.LVUS44:
	.uleb128 .LVU109
	.uleb128 .LVU110
.LLST44:
	.byte	0x8
	.quad	.LVL26
	.uleb128 .LVL26-.LVL26
	.uleb128 0x2
	.byte	0x30
	.byte	0x9f
	.byte	0
.LVUS45:
	.uleb128 0
	.uleb128 .LVU133
	.uleb128 .LVU133
	.uleb128 .LVU174
	.uleb128 .LVU174
	.uleb128 .LVU175
	.uleb128 .LVU175
	.uleb128 0
.LLST45:
	.byte	0x6
	.quad	.LVL34
	.byte	0x4
	.uleb128 .LVL34-.LVL34
	.uleb128 .LVL35-.LVL34
	.uleb128 0x1
	.byte	0x55
	.byte	0x4
	.uleb128 .LVL35-.LVL34
	.uleb128 .LVL50-.LVL34
	.uleb128 0x1
	.byte	0x5d
	.byte	0x4
	.uleb128 .LVL50-.LVL34
	.uleb128 .LVL51-.LVL34
	.uleb128 0x4
	.byte	0xa3
	.uleb128 0x1
	.byte	0x55
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL51-.LVL34
	.uleb128 .LFE5699-.LVL34
	.uleb128 0x1
	.byte	0x5d
	.byte	0
.LVUS46:
	.uleb128 .LVU139
	.uleb128 .LVU145
	.uleb128 .LVU145
	.uleb128 .LVU166
	.uleb128 .LVU175
	.uleb128 .LVU176
.LLST46:
	.byte	0x6
	.quad	.LVL37
	.byte	0x4
	.uleb128 .LVL37-.LVL37
	.uleb128 .LVL38-.LVL37
	.uleb128 0x2
	.byte	0x30
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL38-.LVL37
	.uleb128 .LVL47-.LVL37
	.uleb128 0x1
	.byte	0x56
	.byte	0x4
	.uleb128 .LVL51-.LVL37
	.uleb128 .LVL52-.LVL37
	.uleb128 0x2
	.byte	0x30
	.byte	0x9f
	.byte	0
.LVUS47:
	.uleb128 .LVU139
	.uleb128 .LVU174
	.uleb128 .LVU174
	.uleb128 .LVU175
	.uleb128 .LVU175
	.uleb128 0
.LLST47:
	.byte	0x6
	.quad	.LVL37
	.byte	0x4
	.uleb128 .LVL37-.LVL37
	.uleb128 .LVL50-.LVL37
	.uleb128 0x1
	.byte	0x5d
	.byte	0x4
	.uleb128 .LVL50-.LVL37
	.uleb128 .LVL51-.LVL37
	.uleb128 0x4
	.byte	0xa3
	.uleb128 0x1
	.byte	0x55
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL51-.LVL37
	.uleb128 .LFE5699-.LVL37
	.uleb128 0x1
	.byte	0x5d
	.byte	0
.LVUS48:
	.uleb128 .LVU151
	.uleb128 .LVU153
	.uleb128 .LVU153
	.uleb128 .LVU166
.LLST48:
	.byte	0x6
	.quad	.LVL40
	.byte	0x4
	.uleb128 .LVL40-.LVL40
	.uleb128 .LVL41-1-.LVL40
	.uleb128 0x1
	.byte	0x61
	.byte	0x4
	.uleb128 .LVL41-1-.LVL40
	.uleb128 .LVL47-.LVL40
	.uleb128 0x3
	.byte	0x91
	.sleb128 -72
	.byte	0
.LVUS49:
	.uleb128 .LVU158
	.uleb128 .LVU161
.LLST49:
	.byte	0x8
	.quad	.LVL42
	.uleb128 .LVL43-.LVL42
	.uleb128 0x1
	.byte	0x62
	.byte	0
.LVUS50:
	.uleb128 .LVU159
	.uleb128 .LVU161
	.uleb128 .LVU163
	.uleb128 .LVU166
.LLST50:
	.byte	0x6
	.quad	.LVL42
	.byte	0x4
	.uleb128 .LVL42-.LVL42
	.uleb128 .LVL43-.LVL42
	.uleb128 0x10
	.byte	0xa5
	.uleb128 0x11
	.uleb128 0x2a
	.byte	0xa5
	.uleb128 0x11
	.uleb128 0x2a
	.byte	0x1e
	.byte	0xa5
	.uleb128 0x12
	.uleb128 0x2a
	.byte	0xa5
	.uleb128 0x12
	.uleb128 0x2a
	.byte	0x1e
	.byte	0x22
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL44-.LVL42
	.uleb128 .LVL47-.LVL42
	.uleb128 0x1
	.byte	0x61
	.byte	0
.LVUS52:
	.uleb128 .LVU142
	.uleb128 .LVU145
	.uleb128 .LVU145
	.uleb128 .LVU166
	.uleb128 .LVU175
	.uleb128 .LVU176
.LLST52:
	.byte	0x6
	.quad	.LVL37
	.byte	0x4
	.uleb128 .LVL37-.LVL37
	.uleb128 .LVL38-.LVL37
	.uleb128 0x2
	.byte	0x30
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL38-.LVL37
	.uleb128 .LVL47-.LVL37
	.uleb128 0x1
	.byte	0x53
	.byte	0x4
	.uleb128 .LVL51-.LVL37
	.uleb128 .LVL52-.LVL37
	.uleb128 0x2
	.byte	0x30
	.byte	0x9f
	.byte	0
.Ldebug_loc3:
	.section	.debug_aranges,"",@progbits
	.long	0x3c
	.value	0x2
	.long	.Ldebug_info0
	.byte	0x8
	.byte	0
	.value	0
	.value	0
	.quad	.Ltext0
	.quad	.Letext0-.Ltext0
	.quad	.LFB5701
	.quad	.LFE5701-.LFB5701
	.quad	0
	.quad	0
	.section	.debug_rnglists,"",@progbits
.Ldebug_ranges0:
	.long	.Ldebug_ranges3-.Ldebug_ranges2
.Ldebug_ranges2:
	.value	0x5
	.byte	0x8
	.byte	0
	.long	0
.LLRL15:
	.byte	0x5
	.quad	.LBB48
	.byte	0x4
	.uleb128 .LBB48-.LBB48
	.uleb128 .LBE48-.LBB48
	.byte	0x4
	.uleb128 .LBB85-.LBB48
	.uleb128 .LBE85-.LBB48
	.byte	0
.LLRL17:
	.byte	0x5
	.quad	.LBB49
	.byte	0x4
	.uleb128 .LBB49-.LBB49
	.uleb128 .LBE49-.LBB49
	.byte	0x4
	.uleb128 .LBB52-.LBB49
	.uleb128 .LBE52-.LBB49
	.byte	0
.LLRL21:
	.byte	0x5
	.quad	.LBB55
	.byte	0x4
	.uleb128 .LBB55-.LBB55
	.uleb128 .LBE55-.LBB55
	.byte	0x4
	.uleb128 .LBB58-.LBB55
	.uleb128 .LBE58-.LBB55
	.byte	0
.LLRL24:
	.byte	0x5
	.quad	.LBB59
	.byte	0x4
	.uleb128 .LBB59-.LBB59
	.uleb128 .LBE59-.LBB59
	.byte	0x4
	.uleb128 .LBB65-.LBB59
	.uleb128 .LBE65-.LBB59
	.byte	0
.LLRL26:
	.byte	0x5
	.quad	.LBB62
	.byte	0x4
	.uleb128 .LBB62-.LBB62
	.uleb128 .LBE62-.LBB62
	.byte	0x4
	.uleb128 .LBB84-.LBB62
	.uleb128 .LBE84-.LBB62
	.byte	0
.LLRL31:
	.byte	0x5
	.quad	.LBB68
	.byte	0x4
	.uleb128 .LBB68-.LBB68
	.uleb128 .LBE68-.LBB68
	.byte	0x4
	.uleb128 .LBB77-.LBB68
	.uleb128 .LBE77-.LBB68
	.byte	0
.LLRL34:
	.byte	0x5
	.quad	.LBB71
	.byte	0x4
	.uleb128 .LBB71-.LBB71
	.uleb128 .LBE71-.LBB71
	.byte	0x4
	.uleb128 .LBB78-.LBB71
	.uleb128 .LBE78-.LBB71
	.byte	0
.LLRL36:
	.byte	0x5
	.quad	.LBB74
	.byte	0x4
	.uleb128 .LBB74-.LBB74
	.uleb128 .LBE74-.LBB74
	.byte	0x4
	.uleb128 .LBB79-.LBB74
	.uleb128 .LBE79-.LBB74
	.byte	0
.LLRL43:
	.byte	0x5
	.quad	.LBB86
	.byte	0x4
	.uleb128 .LBB86-.LBB86
	.uleb128 .LBE86-.LBB86
	.byte	0x4
	.uleb128 .LBB87-.LBB86
	.uleb128 .LBE87-.LBB86
	.byte	0x4
	.uleb128 .LBB88-.LBB86
	.uleb128 .LBE88-.LBB86
	.byte	0
.LLRL51:
	.byte	0x5
	.quad	.LBB89
	.byte	0x4
	.uleb128 .LBB89-.LBB89
	.uleb128 .LBE89-.LBB89
	.byte	0x4
	.uleb128 .LBB90-.LBB89
	.uleb128 .LBE90-.LBB89
	.byte	0
.LLRL62:
	.byte	0x5
	.quad	.LBB95
	.byte	0x4
	.uleb128 .LBB95-.LBB95
	.uleb128 .LBE95-.LBB95
	.byte	0x4
	.uleb128 .LBB99-.LBB95
	.uleb128 .LBE99-.LBB95
	.byte	0x4
	.uleb128 .LBB104-.LBB95
	.uleb128 .LBE104-.LBB95
	.byte	0
.LLRL64:
	.byte	0x5
	.quad	.LBB100
	.byte	0x4
	.uleb128 .LBB100-.LBB100
	.uleb128 .LBE100-.LBB100
	.byte	0x4
	.uleb128 .LBB105-.LBB100
	.uleb128 .LBE105-.LBB100
	.byte	0x4
	.uleb128 .LBB106-.LBB100
	.uleb128 .LBE106-.LBB100
	.byte	0
.LLRL68:
	.byte	0x5
	.quad	.LBB109
	.byte	0x4
	.uleb128 .LBB109-.LBB109
	.uleb128 .LBE109-.LBB109
	.byte	0x4
	.uleb128 .LBB114-.LBB109
	.uleb128 .LBE114-.LBB109
	.byte	0x4
	.uleb128 .LBB115-.LBB109
	.uleb128 .LBE115-.LBB109
	.byte	0x4
	.uleb128 .LBB116-.LBB109
	.uleb128 .LBE116-.LBB109
	.byte	0
.LLRL70:
	.byte	0x7
	.quad	.Ltext0
	.uleb128 .Letext0-.Ltext0
	.byte	0x7
	.quad	.LFB5701
	.uleb128 .LFE5701-.LFB5701
	.byte	0
.Ldebug_ranges3:
	.section	.debug_line,"",@progbits
.Ldebug_line0:
	.section	.debug_str,"MS",@progbits,1
.LASF107:
	.string	"printf"
.LASF57:
	.string	"malloc"
.LASF10:
	.string	"size_t"
.LASF68:
	.string	"total_toss"
.LASF32:
	.string	"__align"
.LASF44:
	.string	"uint64_t"
.LASF69:
	.string	"total_thread_toss"
.LASF31:
	.string	"__size"
.LASF73:
	.string	"single_thread_estimation_SIMD"
.LASF16:
	.string	"long long unsigned int"
.LASF18:
	.string	"__next"
.LASF81:
	.string	"x_ps_vec"
.LASF33:
	.string	"pthread_mutexattr_t"
.LASF95:
	.string	"_mm256_set_epi32"
.LASF59:
	.string	"free"
.LASF62:
	.string	"pthread_mutex_lock"
.LASF34:
	.string	"pthread_attr_t"
.LASF14:
	.string	"long long int"
.LASF8:
	.string	"signed char"
.LASF54:
	.string	"pthread_join"
.LASF53:
	.string	"pthread_mutex_destroy"
.LASF3:
	.string	"long int"
.LASF52:
	.string	"strtol"
.LASF113:
	.string	"__builtin_puts"
.LASF2:
	.string	"double"
.LASF51:
	.string	"__printf_chk"
.LASF58:
	.string	"avx_xorshift128plus"
.LASF19:
	.string	"__pthread_list_t"
.LASF50:
	.string	"total_hit"
.LASF17:
	.string	"__prev"
.LASF21:
	.string	"__pthread_mutex_s"
.LASF75:
	.string	"simd_seed"
.LASF97:
	.string	"_mm256_store_si256"
.LASF70:
	.string	"est_pi"
.LASF88:
	.string	"toss"
.LASF77:
	.string	"x_i_vec"
.LASF5:
	.string	"unsigned int"
.LASF87:
	.string	"mask_i_vec"
.LASF90:
	.string	"mykey"
.LASF91:
	.string	"seed"
.LASF4:
	.string	"long unsigned int"
.LASF99:
	.string	"_mm256_cvtepi32_ps"
.LASF26:
	.string	"__kind"
.LASF35:
	.string	"__data"
.LASF63:
	.string	"rand_r"
.LASF28:
	.string	"__elision"
.LASF7:
	.string	"short unsigned int"
.LASF98:
	.string	"_mm256_cvtps_epi32"
.LASF102:
	.string	"_mm256_div_ps"
.LASF96:
	.string	"_mm256_set_ps"
.LASF49:
	.string	"total_hit_mutex"
.LASF100:
	.string	"_mm256_cmp_ps"
.LASF56:
	.string	"pthread_mutex_init"
.LASF24:
	.string	"__owner"
.LASF84:
	.string	"intmax_ps_vec"
.LASF112:
	.string	"puts"
.LASF42:
	.string	"__m256"
.LASF45:
	.string	"avx_xorshift128plus_key_s"
.LASF71:
	.string	"thread_handles"
.LASF48:
	.string	"avx_xorshift128plus_key_t"
.LASF78:
	.string	"y_i_vec"
.LASF104:
	.string	"_mm256_add_ps"
.LASF15:
	.string	"time_t"
.LASF106:
	.string	"__nptr"
.LASF20:
	.string	"__pthread_internal_list"
.LASF11:
	.string	"__uint64_t"
.LASF38:
	.string	"float"
.LASF85:
	.string	"one_ps_vec"
.LASF82:
	.string	"y_ps_vec"
.LASF109:
	.string	"GNU C17 11.3.0 -mavx2 -mtune=generic -march=x86-64 -g -O3 -fasynchronous-unwind-tables -fstack-protector-strong -fstack-clash-protection -fcf-protection"
.LASF110:
	.string	"rand"
.LASF30:
	.string	"pthread_t"
.LASF105:
	.string	"atoi"
.LASF6:
	.string	"unsigned char"
.LASF9:
	.string	"short int"
.LASF39:
	.string	"__v8sf"
.LASF40:
	.string	"__v8si"
.LASF83:
	.string	"dist_ps_vec"
.LASF41:
	.string	"__v8su"
.LASF114:
	.string	"__stack_chk_fail"
.LASF86:
	.string	"cmp_ps_vec"
.LASF23:
	.string	"__count"
.LASF22:
	.string	"__lock"
.LASF79:
	.string	"cmp_i_vec"
.LASF37:
	.string	"long double"
.LASF13:
	.string	"char"
.LASF74:
	.string	"params"
.LASF89:
	.string	"single_thread_estimation_serial"
.LASF80:
	.string	"cnt_i_vec"
.LASF60:
	.string	"avx_xorshift128plus_init"
.LASF101:
	.string	"_mm256_mul_ps"
.LASF108:
	.string	"__fmt"
.LASF36:
	.string	"pthread_mutex_t"
.LASF93:
	.string	"dist"
.LASF64:
	.string	"time"
.LASF27:
	.string	"__spins"
.LASF103:
	.string	"_mm256_and_ps"
.LASF12:
	.string	"__time_t"
.LASF92:
	.string	"num_toss"
.LASF67:
	.string	"num_thread"
.LASF66:
	.string	"argv"
.LASF55:
	.string	"pthread_create"
.LASF94:
	.string	"_mm256_add_epi32"
.LASF25:
	.string	"__nusers"
.LASF43:
	.string	"__m256i"
.LASF46:
	.string	"part1"
.LASF47:
	.string	"part2"
.LASF65:
	.string	"argc"
.LASF76:
	.string	"num_toss_simd"
.LASF29:
	.string	"__list"
.LASF111:
	.string	"sub_hit"
.LASF61:
	.string	"pthread_mutex_unlock"
.LASF72:
	.string	"main"
	.section	.debug_line_str,"MS",@progbits,1
.LASF0:
	.string	"pi.c"
.LASF1:
	.string	"/home/sam/Desktop/2022-Fall-Parallel-Programming/HW2/part1"
	.ident	"GCC: (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
