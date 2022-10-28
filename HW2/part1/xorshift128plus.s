	.file	"xorshift128plus.c"
	.text
.Ltext0:
	.file 0 "/home/sam/Desktop/2022-Fall-Parallel-Programming/HW2/part1" "SIMDxorshift/src//xorshift128plus.c"
	.p2align 4
	.globl	xorshift128plus
	.type	xorshift128plus, @function
xorshift128plus:
.LVL0:
.LFB1:
	.file 1 "SIMDxorshift/src//xorshift128plus.c"
	.loc 1 3 55 view -0
	.cfi_startproc
	.loc 1 3 55 is_stmt 0 view .LVU1
	endbr64
	.loc 1 4 5 is_stmt 1 view .LVU2
	.loc 1 4 14 is_stmt 0 view .LVU3
	movq	(%rdi), %rax
.LVL1:
	.loc 1 5 5 is_stmt 1 view .LVU4
	.loc 1 5 20 is_stmt 0 view .LVU5
	movq	8(%rdi), %rcx
.LVL2:
	.loc 1 6 5 is_stmt 1 view .LVU6
	.loc 1 7 5 view .LVU7
	.loc 1 7 14 is_stmt 0 view .LVU8
	movq	%rax, %rdx
	.loc 1 6 16 view .LVU9
	vmovq	%rcx, %xmm1
	.loc 1 7 14 view .LVU10
	salq	$23, %rdx
	.loc 1 7 8 view .LVU11
	xorq	%rax, %rdx
.LVL3:
	.loc 1 8 5 is_stmt 1 view .LVU12
	.loc 1 8 45 is_stmt 0 view .LVU13
	movq	%rcx, %rax
	shrq	$5, %rax
	.loc 1 8 39 view .LVU14
	xorq	%rcx, %rax
	xorq	%rdx, %rax
	.loc 1 8 32 view .LVU15
	shrq	$18, %rdx
.LVL4:
	.loc 1 8 39 view .LVU16
	xorq	%rdx, %rax
	.loc 1 6 16 view .LVU17
	vpinsrq	$1, %rax, %xmm1, %xmm0
	.loc 1 9 23 view .LVU18
	addq	%rcx, %rax
	.loc 1 6 16 view .LVU19
	vmovdqu	%xmm0, (%rdi)
.LVL5:
	.loc 1 9 5 is_stmt 1 view .LVU20
	.loc 1 10 1 is_stmt 0 view .LVU21
	ret
	.cfi_endproc
.LFE1:
	.size	xorshift128plus, .-xorshift128plus
	.p2align 4
	.globl	xorshift128plus_jump
	.type	xorshift128plus_jump, @function
xorshift128plus_jump:
.LVL6:
.LFB2:
	.loc 1 13 56 is_stmt 1 view -0
	.cfi_startproc
	.loc 1 13 56 is_stmt 0 view .LVU23
	endbr64
	.loc 1 14 5 is_stmt 1 view .LVU24
	.loc 1 15 5 view .LVU25
.LVL7:
	.loc 1 16 5 view .LVU26
	.loc 1 17 5 view .LVU27
.LBB20:
	.loc 1 17 9 view .LVU28
	.loc 1 17 31 view .LVU29
	movq	8(%rdi), %r9
	movq	(%rdi), %r10
.LVL8:
.LBB21:
	.loc 1 18 26 view .LVU30
.LBE21:
.LBE20:
	.loc 1 13 56 is_stmt 0 view .LVU31
	movq	%rdi, %r8
.LBB36:
.LBB34:
	.loc 1 18 17 view .LVU32
	xorl	%ecx, %ecx
	.loc 1 19 17 view .LVU33
	movabsq	$-8476663413540573697, %r11
.LBE34:
.LBE36:
	.loc 1 16 14 view .LVU34
	xorl	%esi, %esi
	.loc 1 15 14 view .LVU35
	xorl	%edi, %edi
.LVL9:
	.loc 1 15 14 view .LVU36
	jmp	.L5
.LVL10:
	.p2align 4,,10
	.p2align 3
.L8:
.LBB37:
.LBB35:
.LBB22:
.LBB23:
	.loc 1 8 39 view .LVU37
	movq	%rax, %r9
.LVL11:
.L5:
	.loc 1 8 39 view .LVU38
.LBE23:
.LBE22:
	.loc 1 19 13 is_stmt 1 view .LVU39
	.loc 1 19 16 is_stmt 0 view .LVU40
	btq	%rcx, %r11
	jnc	.L4
	.loc 1 20 17 is_stmt 1 view .LVU41
	.loc 1 20 20 is_stmt 0 view .LVU42
	xorq	%r10, %rdi
.LVL12:
	.loc 1 21 17 is_stmt 1 view .LVU43
	.loc 1 21 20 is_stmt 0 view .LVU44
	xorq	%r9, %rsi
.LVL13:
.L4:
	.loc 1 23 13 is_stmt 1 view .LVU45
.LBB29:
.LBI22:
	.loc 1 3 10 view .LVU46
.LBB24:
	.loc 1 4 5 view .LVU47
	.loc 1 5 5 view .LVU48
	.loc 1 6 5 view .LVU49
	.loc 1 7 5 view .LVU50
	.loc 1 7 14 is_stmt 0 view .LVU51
	movq	%r10, %rdx
	.loc 1 8 45 view .LVU52
	movq	%r9, %rax
.LBE24:
.LBE29:
	.loc 1 18 33 view .LVU53
	addl	$1, %ecx
.LVL14:
.LBB30:
.LBB25:
	.loc 1 7 14 view .LVU54
	salq	$23, %rdx
	.loc 1 8 45 view .LVU55
	shrq	$5, %rax
	.loc 1 7 8 view .LVU56
	xorq	%r10, %rdx
.LVL15:
	.loc 1 8 5 is_stmt 1 view .LVU57
	movq	%r9, %r10
	.loc 1 8 39 is_stmt 0 view .LVU58
	xorq	%rdx, %rax
	.loc 1 8 32 view .LVU59
	shrq	$18, %rdx
.LVL16:
	.loc 1 8 39 view .LVU60
	xorq	%r9, %rax
	xorq	%rdx, %rax
	.loc 1 9 5 is_stmt 1 view .LVU61
.LVL17:
	.loc 1 9 5 is_stmt 0 view .LVU62
.LBE25:
.LBE30:
	.loc 1 18 33 is_stmt 1 view .LVU63
	.loc 1 18 26 view .LVU64
	cmpl	$64, %ecx
	jne	.L8
	.loc 1 19 17 is_stmt 0 view .LVU65
	movabsq	$1305993406145048470, %r11
	.loc 1 18 17 view .LVU66
	xorl	%r10d, %r10d
	jmp	.L7
.LVL18:
	.p2align 4,,10
	.p2align 3
.L9:
.LBB31:
.LBB26:
	.loc 1 8 39 view .LVU67
	movq	%rdx, %rax
.LVL19:
.L7:
	.loc 1 8 39 view .LVU68
.LBE26:
.LBE31:
	.loc 1 19 13 is_stmt 1 view .LVU69
	.loc 1 19 16 is_stmt 0 view .LVU70
	btq	%r10, %r11
	jnc	.L6
	.loc 1 20 17 is_stmt 1 view .LVU71
	.loc 1 20 20 is_stmt 0 view .LVU72
	xorq	%r9, %rdi
.LVL20:
	.loc 1 21 17 is_stmt 1 view .LVU73
	.loc 1 21 20 is_stmt 0 view .LVU74
	xorq	%rax, %rsi
.LVL21:
.L6:
	.loc 1 23 13 is_stmt 1 view .LVU75
.LBB32:
	.loc 1 3 10 view .LVU76
.LBB27:
	.loc 1 4 5 view .LVU77
	.loc 1 5 5 view .LVU78
	.loc 1 6 5 view .LVU79
	.loc 1 7 5 view .LVU80
	.loc 1 7 14 is_stmt 0 view .LVU81
	movq	%r9, %rcx
	.loc 1 8 45 view .LVU82
	movq	%rax, %rdx
.LBE27:
.LBE32:
	.loc 1 18 33 view .LVU83
	addl	$1, %r10d
.LVL22:
.LBB33:
.LBB28:
	.loc 1 7 14 view .LVU84
	salq	$23, %rcx
	.loc 1 8 45 view .LVU85
	shrq	$5, %rdx
	.loc 1 7 8 view .LVU86
	xorq	%r9, %rcx
.LVL23:
	.loc 1 8 5 is_stmt 1 view .LVU87
	movq	%rax, %r9
	.loc 1 8 39 is_stmt 0 view .LVU88
	xorq	%rcx, %rdx
	.loc 1 8 32 view .LVU89
	shrq	$18, %rcx
.LVL24:
	.loc 1 8 39 view .LVU90
	xorq	%rax, %rdx
	xorq	%rcx, %rdx
	.loc 1 9 5 is_stmt 1 view .LVU91
.LVL25:
	.loc 1 9 5 is_stmt 0 view .LVU92
.LBE28:
.LBE33:
	.loc 1 18 33 is_stmt 1 view .LVU93
	.loc 1 18 26 view .LVU94
	cmpl	$64, %r10d
	jne	.L9
	.loc 1 18 26 is_stmt 0 view .LVU95
.LBE35:
	.loc 1 17 66 is_stmt 1 view .LVU96
.LVL26:
	.loc 1 17 31 view .LVU97
.LBE37:
	.loc 1 26 5 view .LVU98
	.loc 1 26 16 is_stmt 0 view .LVU99
	movq	%rdi, (%r8)
	.loc 1 27 5 is_stmt 1 view .LVU100
	.loc 1 27 16 is_stmt 0 view .LVU101
	movq	%rsi, 8(%r8)
	.loc 1 28 1 view .LVU102
	ret
	.cfi_endproc
.LFE2:
	.size	xorshift128plus_jump, .-xorshift128plus_jump
	.p2align 4
	.globl	xorshift128plus_shuffle32_partial
	.type	xorshift128plus_shuffle32_partial, @function
xorshift128plus_shuffle32_partial:
.LVL27:
.LFB6:
	.loc 1 54 136 is_stmt 1 view -0
	.cfi_startproc
	.loc 1 54 136 is_stmt 0 view .LVU104
	endbr64
	.loc 1 55 5 is_stmt 1 view .LVU105
	.loc 1 56 5 view .LVU106
	.loc 1 57 5 view .LVU107
	testl	%ecx, %ecx
	movl	$1, %eax
	.loc 1 54 136 is_stmt 0 view .LVU108
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	movq	%rsi, %r8
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	cmovne	%ecx, %eax
	movl	%edx, %esi
.LVL28:
	.loc 1 54 136 view .LVU109
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	.loc 1 58 42 view .LVU110
	leal	1(%rax), %r15d
	.loc 1 54 136 view .LVU111
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	.loc 1 54 136 view .LVU112
	movq	%rdi, -24(%rsp)
	movl	%eax, -12(%rsp)
.LVL29:
	.loc 1 58 5 is_stmt 1 view .LVU113
	.loc 1 58 19 view .LVU114
	cmpl	%r15d, %esi
	jbe	.L17
	movq	(%rdi), %rax
.LVL30:
	.loc 1 58 19 is_stmt 0 view .LVU115
	movl	%esi, %ecx
	movq	8(%rdi), %rdi
.LVL31:
	.loc 1 58 19 view .LVU116
	jmp	.L18
.LVL32:
	.p2align 4,,10
	.p2align 3
.L20:
.LBB38:
.LBB39:
.LBB40:
.LBB41:
.LBB42:
	.loc 1 8 39 view .LVU117
	movq	%rdx, %rdi
.LVL33:
.L18:
	.loc 1 8 39 view .LVU118
.LBE42:
.LBE41:
.LBE40:
.LBE39:
	.loc 1 59 6 is_stmt 1 discriminator 3 view .LVU119
.LBB56:
.LBI39:
	.loc 1 36 13 discriminator 3 view .LVU120
.LBB51:
	.loc 1 38 2 discriminator 3 view .LVU121
.LBB47:
.LBI41:
	.loc 1 3 10 discriminator 3 view .LVU122
.LBB43:
	.loc 1 4 5 discriminator 3 view .LVU123
	.loc 1 5 5 discriminator 3 view .LVU124
	.loc 1 6 5 discriminator 3 view .LVU125
	.loc 1 7 5 discriminator 3 view .LVU126
	.loc 1 7 14 is_stmt 0 discriminator 3 view .LVU127
	movq	%rax, %rdx
.LBE43:
.LBE47:
	.loc 1 40 29 discriminator 3 view .LVU128
	leal	-1(%rsi), %r9d
.LBB48:
.LBB44:
	.loc 1 7 14 discriminator 3 view .LVU129
	salq	$23, %rdx
.LBE44:
.LBE48:
.LBE51:
.LBE56:
	.loc 1 60 38 discriminator 3 view .LVU130
	leaq	(%r8,%r9,4), %rbx
.LBB57:
.LBB52:
.LBB49:
.LBB45:
	.loc 1 7 8 discriminator 3 view .LVU131
	xorq	%rax, %rdx
.LVL34:
	.loc 1 8 5 is_stmt 1 discriminator 3 view .LVU132
	.loc 1 8 45 is_stmt 0 discriminator 3 view .LVU133
	movq	%rdi, %rax
.LBE45:
.LBE49:
.LBE52:
.LBE57:
	.loc 1 60 24 discriminator 3 view .LVU134
	movl	(%rbx), %r13d
.LBB58:
.LBB53:
.LBB50:
.LBB46:
	.loc 1 8 45 discriminator 3 view .LVU135
	shrq	$5, %rax
	xorq	%rdx, %rax
	.loc 1 8 32 discriminator 3 view .LVU136
	shrq	$18, %rdx
.LVL35:
	.loc 1 8 32 discriminator 3 view .LVU137
	xorq	%rdi, %rax
	.loc 1 8 39 discriminator 3 view .LVU138
	xorq	%rax, %rdx
	.loc 1 9 5 is_stmt 1 discriminator 3 view .LVU139
	.loc 1 9 23 is_stmt 0 discriminator 3 view .LVU140
	leaq	(%rdi,%rdx), %rax
.LVL36:
	.loc 1 9 23 discriminator 3 view .LVU141
.LBE46:
.LBE50:
	.loc 1 39 2 is_stmt 1 discriminator 3 view .LVU142
	.loc 1 40 2 discriminator 3 view .LVU143
	.loc 1 40 2 is_stmt 0 discriminator 3 view .LVU144
.LBE53:
.LBE58:
	.loc 1 60 9 is_stmt 1 discriminator 3 view .LVU145
	.loc 1 61 9 discriminator 3 view .LVU146
.LBB59:
.LBB54:
	.loc 1 39 22 is_stmt 0 discriminator 3 view .LVU147
	movl	%eax, %r10d
	.loc 1 40 22 discriminator 3 view .LVU148
	shrq	$32, %rax
.LVL37:
	.loc 1 39 46 discriminator 3 view .LVU149
	imulq	%r10, %rcx
	.loc 1 40 29 discriminator 3 view .LVU150
	imulq	%r9, %rax
	.loc 1 39 56 discriminator 3 view .LVU151
	shrq	$32, %rcx
.LBE54:
.LBE59:
	.loc 1 61 38 discriminator 3 view .LVU152
	leaq	(%r8,%rcx,4), %r11
.LBB60:
.LBB55:
	.loc 1 40 39 discriminator 3 view .LVU153
	shrq	$32, %rax
.LBE55:
.LBE60:
	.loc 1 65 34 discriminator 3 view .LVU154
	leal	-2(%rsi), %ecx
	.loc 1 61 24 discriminator 3 view .LVU155
	movl	(%r11), %r14d
.LVL38:
	.loc 1 62 9 is_stmt 1 discriminator 3 view .LVU156
	.loc 1 66 32 is_stmt 0 discriminator 3 view .LVU157
	leaq	(%r8,%rax,4), %r9
.LVL39:
	.loc 1 65 32 discriminator 3 view .LVU158
	leaq	(%r8,%rcx,4), %r10
	.loc 1 65 34 discriminator 3 view .LVU159
	movq	%rcx, %rsi
.LVL40:
	.loc 1 65 34 discriminator 3 view .LVU160
	movq	%rdi, %rax
	.loc 1 62 24 discriminator 3 view .LVU161
	movl	%r14d, (%rbx)
	.loc 1 63 9 is_stmt 1 discriminator 3 view .LVU162
	.loc 1 63 27 is_stmt 0 discriminator 3 view .LVU163
	movl	%r13d, (%r11)
	.loc 1 65 9 is_stmt 1 discriminator 3 view .LVU164
	.loc 1 65 18 is_stmt 0 discriminator 3 view .LVU165
	movl	(%r10), %ebp
.LVL41:
	.loc 1 66 9 is_stmt 1 discriminator 3 view .LVU166
	.loc 1 66 18 is_stmt 0 discriminator 3 view .LVU167
	movl	(%r9), %r12d
.LVL42:
	.loc 1 67 9 is_stmt 1 discriminator 3 view .LVU168
	.loc 1 67 24 is_stmt 0 discriminator 3 view .LVU169
	movl	%r12d, (%r10)
	.loc 1 68 9 is_stmt 1 discriminator 3 view .LVU170
	.loc 1 68 27 is_stmt 0 discriminator 3 view .LVU171
	movl	%ebp, (%r9)
.LBE38:
	.loc 1 58 48 is_stmt 1 discriminator 3 view .LVU172
.LVL43:
	.loc 1 58 19 discriminator 3 view .LVU173
	cmpl	%r15d, %ecx
	ja	.L20
	movq	-24(%rsp), %rax
	vmovq	%rdi, %xmm1
	vpinsrq	$1, %rdx, %xmm1, %xmm0
	vmovdqu	%xmm0, (%rax)
	movl	%r14d, (%rbx)
	movl	%r13d, (%r11)
	movl	%r12d, (%r10)
	movl	%ebp, (%r9)
.LVL44:
.L17:
	.loc 1 71 5 view .LVU174
	.loc 1 71 7 is_stmt 0 view .LVU175
	cmpl	%esi, -12(%rsp)
	jnb	.L21
.LBB61:
	.loc 1 72 6 is_stmt 1 view .LVU176
.LVL45:
.LBB62:
.LBI62:
	.loc 1 43 17 view .LVU177
.LBB63:
	.loc 1 44 2 view .LVU178
.LBB64:
.LBI64:
	.loc 1 3 10 view .LVU179
.LBB65:
	.loc 1 4 5 view .LVU180
	.loc 1 4 14 is_stmt 0 view .LVU181
	movq	-24(%rsp), %rbx
	movq	(%rbx), %rdx
.LVL46:
	.loc 1 5 5 is_stmt 1 view .LVU182
	.loc 1 5 20 is_stmt 0 view .LVU183
	movq	8(%rbx), %rcx
.LVL47:
	.loc 1 6 5 is_stmt 1 view .LVU184
	.loc 1 7 5 view .LVU185
	.loc 1 7 14 is_stmt 0 view .LVU186
	movq	%rdx, %rax
	.loc 1 6 16 view .LVU187
	vmovq	%rcx, %xmm2
	.loc 1 7 14 view .LVU188
	salq	$23, %rax
	.loc 1 7 8 view .LVU189
	xorq	%rax, %rdx
.LVL48:
	.loc 1 8 5 is_stmt 1 view .LVU190
	.loc 1 8 45 is_stmt 0 view .LVU191
	movq	%rcx, %rax
	shrq	$5, %rax
	.loc 1 8 39 view .LVU192
	xorq	%rcx, %rax
	xorq	%rdx, %rax
	.loc 1 8 32 view .LVU193
	shrq	$18, %rdx
.LVL49:
	.loc 1 8 39 view .LVU194
	xorq	%rdx, %rax
.LBE65:
.LBE64:
.LBE63:
.LBE62:
	.loc 1 73 36 view .LVU195
	leal	-1(%rsi), %edx
.LBB74:
.LBB70:
.LBB68:
.LBB66:
	.loc 1 6 16 view .LVU196
	vpinsrq	$1, %rax, %xmm2, %xmm0
.LBE66:
.LBE68:
	.loc 1 45 16 view .LVU197
	addl	%ecx, %eax
.LBE70:
.LBE74:
	.loc 1 73 34 view .LVU198
	leaq	(%r8,%rdx,4), %rdx
.LBB75:
.LBB71:
	.loc 1 45 16 view .LVU199
	movl	%eax, %eax
.LBE71:
.LBE75:
	.loc 1 73 21 view .LVU200
	movl	(%rdx), %edi
.LBB76:
.LBB72:
.LBB69:
.LBB67:
	.loc 1 6 16 view .LVU201
	vmovdqu	%xmm0, (%rbx)
.LVL50:
	.loc 1 9 5 is_stmt 1 view .LVU202
	.loc 1 9 5 is_stmt 0 view .LVU203
.LBE67:
.LBE69:
	.loc 1 45 2 is_stmt 1 view .LVU204
	.loc 1 45 2 is_stmt 0 view .LVU205
.LBE72:
.LBE76:
	.loc 1 73 6 is_stmt 1 view .LVU206
	.loc 1 74 6 view .LVU207
.LBB77:
.LBB73:
	.loc 1 45 40 is_stmt 0 view .LVU208
	imulq	%rsi, %rax
	.loc 1 45 50 view .LVU209
	shrq	$32, %rax
.LBE73:
.LBE77:
	.loc 1 74 34 view .LVU210
	leaq	(%r8,%rax,4), %rax
	.loc 1 74 21 view .LVU211
	movl	(%rax), %ecx
.LVL51:
	.loc 1 75 9 is_stmt 1 view .LVU212
	.loc 1 75 24 is_stmt 0 view .LVU213
	movl	%ecx, (%rdx)
	.loc 1 76 9 is_stmt 1 view .LVU214
	.loc 1 76 26 is_stmt 0 view .LVU215
	movl	%edi, (%rax)
.LVL52:
.L21:
	.loc 1 76 26 view .LVU216
.LBE61:
	.loc 1 79 1 view .LVU217
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
	.cfi_endproc
.LFE6:
	.size	xorshift128plus_shuffle32_partial, .-xorshift128plus_shuffle32_partial
	.p2align 4
	.globl	xorshift128plus_shuffle32
	.type	xorshift128plus_shuffle32, @function
xorshift128plus_shuffle32:
.LVL53:
.LFB5:
	.loc 1 50 96 is_stmt 1 view -0
	.cfi_startproc
	.loc 1 50 96 is_stmt 0 view .LVU219
	endbr64
	.loc 1 51 5 is_stmt 1 view .LVU220
	movl	$1, %ecx
	jmp	xorshift128plus_shuffle32_partial
.LVL54:
	.loc 1 51 5 is_stmt 0 view .LVU221
	.cfi_endproc
.LFE5:
	.size	xorshift128plus_shuffle32, .-xorshift128plus_shuffle32
.Letext0:
	.file 2 "/usr/include/x86_64-linux-gnu/bits/types.h"
	.file 3 "/usr/include/x86_64-linux-gnu/bits/stdint-uintn.h"
	.file 4 "SIMDxorshift/include/xorshift128plus.h"
	.section	.debug_info,"",@progbits
.Ldebug_info0:
	.long	0x5ae
	.value	0x5
	.byte	0x1
	.byte	0x8
	.long	.Ldebug_abbrev0
	.uleb128 0x14
	.long	.LASF36
	.byte	0x1d
	.long	.LASF0
	.long	.LASF1
	.quad	.Ltext0
	.quad	.Letext0-.Ltext0
	.long	.Ldebug_line0
	.uleb128 0x3
	.byte	0x1
	.byte	0x8
	.long	.LASF2
	.uleb128 0x3
	.byte	0x2
	.byte	0x7
	.long	.LASF3
	.uleb128 0x3
	.byte	0x4
	.byte	0x7
	.long	.LASF4
	.uleb128 0x3
	.byte	0x8
	.byte	0x7
	.long	.LASF5
	.uleb128 0x3
	.byte	0x1
	.byte	0x6
	.long	.LASF6
	.uleb128 0x3
	.byte	0x2
	.byte	0x5
	.long	.LASF7
	.uleb128 0x15
	.byte	0x4
	.byte	0x5
	.string	"int"
	.uleb128 0x7
	.long	.LASF9
	.byte	0x2
	.byte	0x2a
	.byte	0x16
	.long	0x3c
	.uleb128 0x3
	.byte	0x8
	.byte	0x5
	.long	.LASF8
	.uleb128 0x7
	.long	.LASF10
	.byte	0x2
	.byte	0x2d
	.byte	0x1b
	.long	0x43
	.uleb128 0x3
	.byte	0x1
	.byte	0x6
	.long	.LASF11
	.uleb128 0x7
	.long	.LASF12
	.byte	0x3
	.byte	0x1a
	.byte	0x14
	.long	0x5f
	.uleb128 0xc
	.long	0x85
	.uleb128 0x7
	.long	.LASF13
	.byte	0x3
	.byte	0x1b
	.byte	0x14
	.long	0x72
	.uleb128 0xc
	.long	0x96
	.uleb128 0x16
	.long	.LASF37
	.byte	0x10
	.byte	0x4
	.byte	0x9
	.byte	0x8
	.long	0xcb
	.uleb128 0x11
	.long	.LASF14
	.byte	0xa
	.long	0x96
	.byte	0
	.uleb128 0x11
	.long	.LASF15
	.byte	0xb
	.long	0x96
	.byte	0x8
	.byte	0
	.uleb128 0x7
	.long	.LASF16
	.byte	0x4
	.byte	0xe
	.byte	0x26
	.long	0xa7
	.uleb128 0xd
	.long	.LASF26
	.byte	0x36
	.byte	0x7
	.quad	.LFB6
	.quad	.LFE6-.LFB6
	.uleb128 0x1
	.byte	0x9c
	.long	0x344
	.uleb128 0xe
	.string	"key"
	.byte	0x36
	.byte	0x41
	.long	0x344
	.long	.LLST13
	.long	.LVUS13
	.uleb128 0x8
	.long	.LASF17
	.byte	0x36
	.byte	0x50
	.long	0x349
	.long	.LLST14
	.long	.LVUS14
	.uleb128 0x8
	.long	.LASF18
	.byte	0x36
	.byte	0x62
	.long	0x85
	.long	.LLST15
	.long	.LVUS15
	.uleb128 0x8
	.long	.LASF19
	.byte	0x36
	.byte	0x71
	.long	0x85
	.long	.LLST16
	.long	.LVUS16
	.uleb128 0x4
	.string	"i"
	.byte	0x37
	.byte	0xe
	.long	0x85
	.long	.LLST17
	.long	.LVUS17
	.uleb128 0x6
	.long	.LASF20
	.byte	0x38
	.byte	0xe
	.long	0x85
	.long	.LLST18
	.long	.LVUS18
	.uleb128 0x6
	.long	.LASF21
	.byte	0x38
	.byte	0x18
	.long	0x85
	.long	.LLST19
	.long	.LVUS19
	.uleb128 0x17
	.quad	.LBB38
	.quad	.LBE38-.LBB38
	.long	0x284
	.uleb128 0x6
	.long	.LASF22
	.byte	0x3c
	.byte	0x18
	.long	0x91
	.long	.LLST20
	.long	.LVUS20
	.uleb128 0x6
	.long	.LASF23
	.byte	0x3d
	.byte	0x18
	.long	0x91
	.long	.LLST21
	.long	.LVUS21
	.uleb128 0x6
	.long	.LASF24
	.byte	0x41
	.byte	0x12
	.long	0x85
	.long	.LLST22
	.long	.LVUS22
	.uleb128 0x6
	.long	.LASF25
	.byte	0x42
	.byte	0x12
	.long	0x85
	.long	.LLST23
	.long	.LVUS23
	.uleb128 0x9
	.long	0x400
	.quad	.LBI39
	.byte	.LVU120
	.long	.LLRL24
	.byte	0x3b
	.byte	0x6
	.uleb128 0x1
	.long	0x439
	.long	.LLST25
	.long	.LVUS25
	.uleb128 0x1
	.long	0x42e
	.long	.LLST26
	.long	.LVUS26
	.uleb128 0x1
	.long	0x423
	.long	.LLST27
	.long	.LVUS27
	.uleb128 0x1
	.long	0x418
	.long	.LLST28
	.long	.LVUS28
	.uleb128 0x1
	.long	0x40d
	.long	.LLST29
	.long	.LVUS29
	.uleb128 0x5
	.long	.LLRL24
	.uleb128 0x2
	.long	0x444
	.long	.LLST30
	.long	.LVUS30
	.uleb128 0x9
	.long	0x547
	.quad	.LBI41
	.byte	.LVU122
	.long	.LLRL31
	.byte	0x26
	.byte	0x12
	.uleb128 0x1
	.long	0x558
	.long	.LLST32
	.long	.LVUS32
	.uleb128 0x5
	.long	.LLRL31
	.uleb128 0x2
	.long	0x563
	.long	.LLST33
	.long	.LVUS33
	.uleb128 0x2
	.long	0x56d
	.long	.LLST34
	.long	.LVUS34
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x18
	.quad	.LBB61
	.quad	.LBE61-.LBB61
	.uleb128 0xf
	.long	.LASF29
	.byte	0x48
	.byte	0x15
	.long	0x91
	.uleb128 0x4
	.string	"tmp"
	.byte	0x49
	.byte	0x15
	.long	0x91
	.long	.LLST35
	.long	.LVUS35
	.uleb128 0x4
	.string	"val"
	.byte	0x4a
	.byte	0x15
	.long	0x91
	.long	.LLST36
	.long	.LVUS36
	.uleb128 0x9
	.long	0x3cd
	.quad	.LBI62
	.byte	.LVU177
	.long	.LLRL37
	.byte	0x48
	.byte	0x1f
	.uleb128 0x1
	.long	0x3e9
	.long	.LLST38
	.long	.LVUS38
	.uleb128 0x1
	.long	0x3de
	.long	.LLST39
	.long	.LVUS39
	.uleb128 0x5
	.long	.LLRL37
	.uleb128 0x19
	.long	0x3f4
	.uleb128 0x9
	.long	0x547
	.quad	.LBI64
	.byte	.LVU179
	.long	.LLRL40
	.byte	0x2c
	.byte	0x12
	.uleb128 0x1
	.long	0x558
	.long	.LLST41
	.long	.LVUS41
	.uleb128 0x5
	.long	.LLRL40
	.uleb128 0x2
	.long	0x563
	.long	.LLST42
	.long	.LVUS42
	.uleb128 0x2
	.long	0x56d
	.long	.LLST43
	.long	.LVUS43
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x12
	.long	0xcb
	.uleb128 0x12
	.long	0x85
	.uleb128 0xd
	.long	.LASF27
	.byte	0x32
	.byte	0x7
	.quad	.LFB5
	.quad	.LFE5-.LFB5
	.uleb128 0x1
	.byte	0x9c
	.long	0x3cd
	.uleb128 0xe
	.string	"key"
	.byte	0x32
	.byte	0x39
	.long	0x344
	.long	.LLST44
	.long	.LVUS44
	.uleb128 0x8
	.long	.LASF17
	.byte	0x32
	.byte	0x48
	.long	0x349
	.long	.LLST45
	.long	.LVUS45
	.uleb128 0x8
	.long	.LASF18
	.byte	0x32
	.byte	0x5a
	.long	0x85
	.long	.LLST46
	.long	.LVUS46
	.uleb128 0x1a
	.quad	.LVL54
	.long	0xd7
	.uleb128 0xb
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x3
	.byte	0xa3
	.uleb128 0x1
	.byte	0x55
	.uleb128 0xb
	.uleb128 0x1
	.byte	0x54
	.uleb128 0x3
	.byte	0xa3
	.uleb128 0x1
	.byte	0x54
	.uleb128 0xb
	.uleb128 0x1
	.byte	0x51
	.uleb128 0x3
	.byte	0xa3
	.uleb128 0x1
	.byte	0x51
	.uleb128 0xb
	.uleb128 0x1
	.byte	0x52
	.uleb128 0x1
	.byte	0x31
	.byte	0
	.byte	0
	.uleb128 0x1b
	.long	.LASF38
	.byte	0x1
	.byte	0x2b
	.byte	0x11
	.long	0x85
	.byte	0x1
	.long	0x400
	.uleb128 0x10
	.string	"key"
	.byte	0x2b
	.byte	0x41
	.long	0x344
	.uleb128 0xa
	.long	.LASF28
	.byte	0x2b
	.byte	0x4f
	.long	0x85
	.uleb128 0xf
	.long	.LASF30
	.byte	0x2c
	.byte	0xb
	.long	0x96
	.byte	0
	.uleb128 0x1c
	.long	.LASF39
	.byte	0x1
	.byte	0x24
	.byte	0xd
	.byte	0x1
	.long	0x450
	.uleb128 0x10
	.string	"key"
	.byte	0x24
	.byte	0x48
	.long	0x344
	.uleb128 0xa
	.long	.LASF31
	.byte	0x25
	.byte	0xc
	.long	0x85
	.uleb128 0xa
	.long	.LASF32
	.byte	0x25
	.byte	0x1d
	.long	0x85
	.uleb128 0xa
	.long	.LASF33
	.byte	0x25
	.byte	0x2f
	.long	0x349
	.uleb128 0xa
	.long	.LASF34
	.byte	0x25
	.byte	0x44
	.long	0x349
	.uleb128 0xf
	.long	.LASF30
	.byte	0x26
	.byte	0xb
	.long	0x96
	.byte	0
	.uleb128 0xd
	.long	.LASF35
	.byte	0xd
	.byte	0x6
	.quad	.LFB2
	.quad	.LFE2-.LFB2
	.uleb128 0x1
	.byte	0x9c
	.long	0x532
	.uleb128 0xe
	.string	"key"
	.byte	0xd
	.byte	0x33
	.long	0x344
	.long	.LLST2
	.long	.LVUS2
	.uleb128 0x1d
	.long	.LASF40
	.byte	0x1
	.byte	0xe
	.byte	0x1b
	.long	0x542
	.byte	0x10
	.byte	0xff
	.byte	0x2d
	.byte	0x5d
	.byte	0x63
	.byte	0x89
	.byte	0xd7
	.byte	0x5c
	.byte	0x8a
	.byte	0x96
	.byte	0x2f
	.byte	0x47
	.byte	0x5c
	.byte	0x15
	.byte	0xd2
	.byte	0x1f
	.byte	0x12
	.uleb128 0x4
	.string	"s0"
	.byte	0xf
	.byte	0xe
	.long	0x96
	.long	.LLST3
	.long	.LVUS3
	.uleb128 0x4
	.string	"s1"
	.byte	0x10
	.byte	0xe
	.long	0x96
	.long	.LLST4
	.long	.LVUS4
	.uleb128 0x5
	.long	.LLRL5
	.uleb128 0x4
	.string	"i"
	.byte	0x11
	.byte	0x16
	.long	0x3c
	.long	.LLST6
	.long	.LVUS6
	.uleb128 0x5
	.long	.LLRL7
	.uleb128 0x4
	.string	"b"
	.byte	0x12
	.byte	0x11
	.long	0x58
	.long	.LLST8
	.long	.LVUS8
	.uleb128 0x9
	.long	0x547
	.quad	.LBI22
	.byte	.LVU46
	.long	.LLRL9
	.byte	0x17
	.byte	0xd
	.uleb128 0x1
	.long	0x558
	.long	.LLST10
	.long	.LVUS10
	.uleb128 0x5
	.long	.LLRL9
	.uleb128 0x2
	.long	0x563
	.long	.LLST11
	.long	.LVUS11
	.uleb128 0x2
	.long	0x56d
	.long	.LLST12
	.long	.LVUS12
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.uleb128 0x1e
	.long	0xa2
	.long	0x542
	.uleb128 0x1f
	.long	0x43
	.byte	0x1
	.byte	0
	.uleb128 0xc
	.long	0x532
	.uleb128 0x20
	.long	.LASF41
	.byte	0x1
	.byte	0x3
	.byte	0xa
	.long	0x96
	.byte	0x1
	.long	0x578
	.uleb128 0x10
	.string	"key"
	.byte	0x3
	.byte	0x32
	.long	0x344
	.uleb128 0x13
	.string	"s1"
	.byte	0x4
	.byte	0xe
	.long	0x96
	.uleb128 0x13
	.string	"s0"
	.byte	0x5
	.byte	0x14
	.long	0xa2
	.byte	0
	.uleb128 0x21
	.long	0x547
	.quad	.LFB1
	.quad	.LFE1-.LFB1
	.uleb128 0x1
	.byte	0x9c
	.uleb128 0x22
	.long	0x558
	.uleb128 0x1
	.byte	0x55
	.uleb128 0x2
	.long	0x563
	.long	.LLST0
	.long	.LVUS0
	.uleb128 0x2
	.long	0x56d
	.long	.LLST1
	.long	.LVUS1
	.byte	0
	.byte	0
	.section	.debug_abbrev,"",@progbits
.Ldebug_abbrev0:
	.uleb128 0x1
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
	.uleb128 0x2
	.uleb128 0x34
	.byte	0
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x17
	.uleb128 0x2137
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x3
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
	.uleb128 0x4
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
	.uleb128 0x5
	.uleb128 0xb
	.byte	0x1
	.uleb128 0x55
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x6
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
	.uleb128 0x7
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
	.uleb128 0x8
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
	.uleb128 0x9
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
	.byte	0
	.byte	0
	.uleb128 0xa
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
	.byte	0
	.byte	0
	.uleb128 0xb
	.uleb128 0x49
	.byte	0
	.uleb128 0x2
	.uleb128 0x18
	.uleb128 0x7e
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0xc
	.uleb128 0x26
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0xd
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
	.uleb128 0xe
	.uleb128 0x5
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
	.uleb128 0xf
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
	.byte	0
	.byte	0
	.uleb128 0x10
	.uleb128 0x5
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
	.byte	0
	.byte	0
	.uleb128 0x11
	.uleb128 0xd
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0x21
	.sleb128 4
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0x21
	.sleb128 14
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x38
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x12
	.uleb128 0xf
	.byte	0
	.uleb128 0xb
	.uleb128 0x21
	.sleb128 8
	.uleb128 0x49
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x13
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
	.byte	0
	.byte	0
	.uleb128 0x14
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
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x10
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x15
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
	.uleb128 0x16
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
	.uleb128 0x17
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
	.uleb128 0x18
	.uleb128 0xb
	.byte	0x1
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.byte	0
	.byte	0
	.uleb128 0x19
	.uleb128 0x34
	.byte	0
	.uleb128 0x31
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x1a
	.uleb128 0x48
	.byte	0x1
	.uleb128 0x7d
	.uleb128 0x1
	.uleb128 0x82
	.uleb128 0x19
	.uleb128 0x7f
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x1b
	.uleb128 0x2e
	.byte	0x1
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
	.uleb128 0x20
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x1c
	.uleb128 0x2e
	.byte	0x1
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
	.uleb128 0x20
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x1d
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
	.uleb128 0x1c
	.uleb128 0xa
	.byte	0
	.byte	0
	.uleb128 0x1e
	.uleb128 0x1
	.byte	0x1
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x1f
	.uleb128 0x21
	.byte	0
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x2f
	.uleb128 0xb
	.byte	0
	.byte	0
	.uleb128 0x20
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
	.uleb128 0x20
	.uleb128 0xb
	.uleb128 0x1
	.uleb128 0x13
	.byte	0
	.byte	0
	.uleb128 0x21
	.uleb128 0x2e
	.byte	0x1
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x11
	.uleb128 0x1
	.uleb128 0x12
	.uleb128 0x7
	.uleb128 0x40
	.uleb128 0x18
	.uleb128 0x7a
	.uleb128 0x19
	.byte	0
	.byte	0
	.uleb128 0x22
	.uleb128 0x5
	.byte	0
	.uleb128 0x31
	.uleb128 0x13
	.uleb128 0x2
	.uleb128 0x18
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
.LVUS13:
	.uleb128 0
	.uleb128 .LVU116
	.uleb128 .LVU116
	.uleb128 0
.LLST13:
	.byte	0x4
	.uleb128 .LVL27-.Ltext0
	.uleb128 .LVL31-.Ltext0
	.uleb128 0x1
	.byte	0x55
	.byte	0x4
	.uleb128 .LVL31-.Ltext0
	.uleb128 .LFE6-.Ltext0
	.uleb128 0x3
	.byte	0x91
	.sleb128 -80
	.byte	0
.LVUS14:
	.uleb128 0
	.uleb128 .LVU109
	.uleb128 .LVU109
	.uleb128 0
.LLST14:
	.byte	0x4
	.uleb128 .LVL27-.Ltext0
	.uleb128 .LVL28-.Ltext0
	.uleb128 0x1
	.byte	0x54
	.byte	0x4
	.uleb128 .LVL28-.Ltext0
	.uleb128 .LFE6-.Ltext0
	.uleb128 0x1
	.byte	0x58
	.byte	0
.LVUS15:
	.uleb128 0
	.uleb128 .LVU117
	.uleb128 .LVU117
	.uleb128 0
.LLST15:
	.byte	0x4
	.uleb128 .LVL27-.Ltext0
	.uleb128 .LVL32-.Ltext0
	.uleb128 0x1
	.byte	0x51
	.byte	0x4
	.uleb128 .LVL32-.Ltext0
	.uleb128 .LFE6-.Ltext0
	.uleb128 0x4
	.byte	0xa3
	.uleb128 0x1
	.byte	0x51
	.byte	0x9f
	.byte	0
.LVUS16:
	.uleb128 0
	.uleb128 .LVU113
	.uleb128 .LVU113
	.uleb128 .LVU115
	.uleb128 .LVU115
	.uleb128 0
.LLST16:
	.byte	0x4
	.uleb128 .LVL27-.Ltext0
	.uleb128 .LVL29-.Ltext0
	.uleb128 0x1
	.byte	0x52
	.byte	0x4
	.uleb128 .LVL29-.Ltext0
	.uleb128 .LVL30-.Ltext0
	.uleb128 0x1
	.byte	0x50
	.byte	0x4
	.uleb128 .LVL30-.Ltext0
	.uleb128 .LFE6-.Ltext0
	.uleb128 0x3
	.byte	0x91
	.sleb128 -68
	.byte	0
.LVUS17:
	.uleb128 .LVU114
	.uleb128 .LVU117
	.uleb128 .LVU117
	.uleb128 .LVU118
	.uleb128 .LVU118
	.uleb128 .LVU160
	.uleb128 .LVU173
	.uleb128 .LVU174
.LLST17:
	.byte	0x4
	.uleb128 .LVL29-.Ltext0
	.uleb128 .LVL32-.Ltext0
	.uleb128 0x1
	.byte	0x54
	.byte	0x4
	.uleb128 .LVL32-.Ltext0
	.uleb128 .LVL33-.Ltext0
	.uleb128 0x1
	.byte	0x52
	.byte	0x4
	.uleb128 .LVL33-.Ltext0
	.uleb128 .LVL40-.Ltext0
	.uleb128 0x1
	.byte	0x54
	.byte	0x4
	.uleb128 .LVL43-.Ltext0
	.uleb128 .LVL44-.Ltext0
	.uleb128 0x1
	.byte	0x52
	.byte	0
.LVUS18:
	.uleb128 .LVU143
	.uleb128 .LVU149
	.uleb128 .LVU149
	.uleb128 .LVU160
.LLST18:
	.byte	0x4
	.uleb128 .LVL36-.Ltext0
	.uleb128 .LVL37-.Ltext0
	.uleb128 0x15
	.byte	0x70
	.sleb128 0
	.byte	0xc
	.long	0xffffffff
	.byte	0x1a
	.byte	0x74
	.sleb128 0
	.byte	0xc
	.long	0xffffffff
	.byte	0x1a
	.byte	0x1e
	.byte	0x8
	.byte	0x20
	.byte	0x25
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL37-.Ltext0
	.uleb128 .LVL40-.Ltext0
	.uleb128 0x18
	.byte	0x75
	.sleb128 0
	.byte	0x71
	.sleb128 0
	.byte	0x22
	.byte	0xc
	.long	0xffffffff
	.byte	0x1a
	.byte	0x74
	.sleb128 0
	.byte	0xc
	.long	0xffffffff
	.byte	0x1a
	.byte	0x1e
	.byte	0x8
	.byte	0x20
	.byte	0x25
	.byte	0x9f
	.byte	0
.LVUS19:
	.uleb128 .LVU144
	.uleb128 .LVU149
	.uleb128 .LVU149
	.uleb128 .LVU158
	.uleb128 .LVU158
	.uleb128 .LVU160
.LLST19:
	.byte	0x4
	.uleb128 .LVL36-.Ltext0
	.uleb128 .LVL37-.Ltext0
	.uleb128 0xc
	.byte	0x70
	.sleb128 0
	.byte	0x8
	.byte	0x20
	.byte	0x25
	.byte	0x79
	.sleb128 0
	.byte	0x1e
	.byte	0x8
	.byte	0x20
	.byte	0x25
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL37-.Ltext0
	.uleb128 .LVL39-.Ltext0
	.uleb128 0xf
	.byte	0x75
	.sleb128 0
	.byte	0x71
	.sleb128 0
	.byte	0x22
	.byte	0x8
	.byte	0x20
	.byte	0x25
	.byte	0x79
	.sleb128 0
	.byte	0x1e
	.byte	0x8
	.byte	0x20
	.byte	0x25
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL39-.Ltext0
	.uleb128 .LVL40-.Ltext0
	.uleb128 0x15
	.byte	0x75
	.sleb128 0
	.byte	0x71
	.sleb128 0
	.byte	0x22
	.byte	0x8
	.byte	0x20
	.byte	0x25
	.byte	0x74
	.sleb128 -1
	.byte	0xc
	.long	0xffffffff
	.byte	0x1a
	.byte	0x1e
	.byte	0x8
	.byte	0x20
	.byte	0x25
	.byte	0x9f
	.byte	0
.LVUS20:
	.uleb128 .LVU117
	.uleb128 .LVU118
	.uleb128 .LVU146
	.uleb128 .LVU174
.LLST20:
	.byte	0x4
	.uleb128 .LVL32-.Ltext0
	.uleb128 .LVL33-.Ltext0
	.uleb128 0x1
	.byte	0x5d
	.byte	0x4
	.uleb128 .LVL36-.Ltext0
	.uleb128 .LVL44-.Ltext0
	.uleb128 0x1
	.byte	0x5d
	.byte	0
.LVUS21:
	.uleb128 .LVU117
	.uleb128 .LVU118
	.uleb128 .LVU156
	.uleb128 .LVU174
.LLST21:
	.byte	0x4
	.uleb128 .LVL32-.Ltext0
	.uleb128 .LVL33-.Ltext0
	.uleb128 0x1
	.byte	0x5e
	.byte	0x4
	.uleb128 .LVL38-.Ltext0
	.uleb128 .LVL44-.Ltext0
	.uleb128 0x1
	.byte	0x5e
	.byte	0
.LVUS22:
	.uleb128 .LVU117
	.uleb128 .LVU118
	.uleb128 .LVU166
	.uleb128 .LVU174
.LLST22:
	.byte	0x4
	.uleb128 .LVL32-.Ltext0
	.uleb128 .LVL33-.Ltext0
	.uleb128 0x1
	.byte	0x56
	.byte	0x4
	.uleb128 .LVL41-.Ltext0
	.uleb128 .LVL44-.Ltext0
	.uleb128 0x1
	.byte	0x56
	.byte	0
.LVUS23:
	.uleb128 .LVU117
	.uleb128 .LVU118
	.uleb128 .LVU168
	.uleb128 .LVU174
.LLST23:
	.byte	0x4
	.uleb128 .LVL32-.Ltext0
	.uleb128 .LVL33-.Ltext0
	.uleb128 0x1
	.byte	0x5c
	.byte	0x4
	.uleb128 .LVL42-.Ltext0
	.uleb128 .LVL44-.Ltext0
	.uleb128 0x1
	.byte	0x5c
	.byte	0
.LVUS25:
	.uleb128 .LVU120
	.uleb128 .LVU144
.LLST25:
	.byte	0x4
	.uleb128 .LVL33-.Ltext0
	.uleb128 .LVL36-.Ltext0
	.uleb128 0x6
	.byte	0xa0
	.long	.Ldebug_info0+356
	.sleb128 0
	.byte	0
.LVUS26:
	.uleb128 .LVU120
	.uleb128 .LVU144
.LLST26:
	.byte	0x4
	.uleb128 .LVL33-.Ltext0
	.uleb128 .LVL36-.Ltext0
	.uleb128 0x6
	.byte	0xa0
	.long	.Ldebug_info0+337
	.sleb128 0
	.byte	0
.LVUS27:
	.uleb128 .LVU120
	.uleb128 .LVU144
.LLST27:
	.byte	0x4
	.uleb128 .LVL33-.Ltext0
	.uleb128 .LVL36-.Ltext0
	.uleb128 0x3
	.byte	0x74
	.sleb128 -1
	.byte	0x9f
	.byte	0
.LVUS28:
	.uleb128 .LVU120
	.uleb128 .LVU144
.LLST28:
	.byte	0x4
	.uleb128 .LVL33-.Ltext0
	.uleb128 .LVL36-.Ltext0
	.uleb128 0x1
	.byte	0x54
	.byte	0
.LVUS29:
	.uleb128 .LVU120
	.uleb128 .LVU144
.LLST29:
	.byte	0x4
	.uleb128 .LVL33-.Ltext0
	.uleb128 .LVL36-.Ltext0
	.uleb128 0x3
	.byte	0x91
	.sleb128 -80
	.byte	0
.LVUS30:
	.uleb128 .LVU117
	.uleb128 .LVU118
	.uleb128 .LVU141
	.uleb128 .LVU149
	.uleb128 .LVU149
	.uleb128 .LVU174
.LLST30:
	.byte	0x4
	.uleb128 .LVL32-.Ltext0
	.uleb128 .LVL33-.Ltext0
	.uleb128 0x6
	.byte	0x70
	.sleb128 0
	.byte	0x71
	.sleb128 0
	.byte	0x22
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL36-.Ltext0
	.uleb128 .LVL37-.Ltext0
	.uleb128 0x1
	.byte	0x50
	.byte	0x4
	.uleb128 .LVL37-.Ltext0
	.uleb128 .LVL44-.Ltext0
	.uleb128 0x6
	.byte	0x75
	.sleb128 0
	.byte	0x71
	.sleb128 0
	.byte	0x22
	.byte	0x9f
	.byte	0
.LVUS32:
	.uleb128 .LVU122
	.uleb128 .LVU141
.LLST32:
	.byte	0x4
	.uleb128 .LVL33-.Ltext0
	.uleb128 .LVL36-.Ltext0
	.uleb128 0x3
	.byte	0x91
	.sleb128 -80
	.byte	0
.LVUS33:
	.uleb128 .LVU124
	.uleb128 .LVU132
	.uleb128 .LVU132
	.uleb128 .LVU137
.LLST33:
	.byte	0x4
	.uleb128 .LVL33-.Ltext0
	.uleb128 .LVL34-.Ltext0
	.uleb128 0x1
	.byte	0x50
	.byte	0x4
	.uleb128 .LVL34-.Ltext0
	.uleb128 .LVL35-.Ltext0
	.uleb128 0x1
	.byte	0x51
	.byte	0
.LVUS34:
	.uleb128 .LVU125
	.uleb128 .LVU141
.LLST34:
	.byte	0x4
	.uleb128 .LVL33-.Ltext0
	.uleb128 .LVL36-.Ltext0
	.uleb128 0x1
	.byte	0x55
	.byte	0
.LVUS35:
	.uleb128 .LVU207
	.uleb128 .LVU216
.LLST35:
	.byte	0x4
	.uleb128 .LVL50-.Ltext0
	.uleb128 .LVL52-.Ltext0
	.uleb128 0x1
	.byte	0x55
	.byte	0
.LVUS36:
	.uleb128 .LVU212
	.uleb128 .LVU216
.LLST36:
	.byte	0x4
	.uleb128 .LVL51-.Ltext0
	.uleb128 .LVL52-.Ltext0
	.uleb128 0x1
	.byte	0x52
	.byte	0
.LVUS38:
	.uleb128 .LVU177
	.uleb128 .LVU205
.LLST38:
	.byte	0x4
	.uleb128 .LVL45-.Ltext0
	.uleb128 .LVL50-.Ltext0
	.uleb128 0x1
	.byte	0x54
	.byte	0
.LVUS39:
	.uleb128 .LVU177
	.uleb128 .LVU205
.LLST39:
	.byte	0x4
	.uleb128 .LVL45-.Ltext0
	.uleb128 .LVL50-.Ltext0
	.uleb128 0x3
	.byte	0x91
	.sleb128 -80
	.byte	0
.LVUS41:
	.uleb128 .LVU179
	.uleb128 .LVU203
.LLST41:
	.byte	0x4
	.uleb128 .LVL45-.Ltext0
	.uleb128 .LVL50-.Ltext0
	.uleb128 0x3
	.byte	0x91
	.sleb128 -80
	.byte	0
.LVUS42:
	.uleb128 .LVU182
	.uleb128 .LVU194
	.uleb128 .LVU194
	.uleb128 .LVU202
.LLST42:
	.byte	0x4
	.uleb128 .LVL46-.Ltext0
	.uleb128 .LVL49-.Ltext0
	.uleb128 0x1
	.byte	0x51
	.byte	0x4
	.uleb128 .LVL49-.Ltext0
	.uleb128 .LVL50-.Ltext0
	.uleb128 0xe
	.byte	0x91
	.sleb128 -80
	.byte	0x6
	.byte	0x6
	.byte	0x47
	.byte	0x24
	.byte	0x91
	.sleb128 -80
	.byte	0x6
	.byte	0x6
	.byte	0x27
	.byte	0x9f
	.byte	0
.LVUS43:
	.uleb128 .LVU184
	.uleb128 .LVU203
.LLST43:
	.byte	0x4
	.uleb128 .LVL47-.Ltext0
	.uleb128 .LVL50-.Ltext0
	.uleb128 0x1
	.byte	0x52
	.byte	0
.LVUS44:
	.uleb128 0
	.uleb128 .LVU221
	.uleb128 .LVU221
	.uleb128 0
.LLST44:
	.byte	0x4
	.uleb128 .LVL53-.Ltext0
	.uleb128 .LVL54-1-.Ltext0
	.uleb128 0x1
	.byte	0x55
	.byte	0x4
	.uleb128 .LVL54-1-.Ltext0
	.uleb128 .LFE5-.Ltext0
	.uleb128 0x4
	.byte	0xa3
	.uleb128 0x1
	.byte	0x55
	.byte	0x9f
	.byte	0
.LVUS45:
	.uleb128 0
	.uleb128 .LVU221
	.uleb128 .LVU221
	.uleb128 0
.LLST45:
	.byte	0x4
	.uleb128 .LVL53-.Ltext0
	.uleb128 .LVL54-1-.Ltext0
	.uleb128 0x1
	.byte	0x54
	.byte	0x4
	.uleb128 .LVL54-1-.Ltext0
	.uleb128 .LFE5-.Ltext0
	.uleb128 0x4
	.byte	0xa3
	.uleb128 0x1
	.byte	0x54
	.byte	0x9f
	.byte	0
.LVUS46:
	.uleb128 0
	.uleb128 .LVU221
	.uleb128 .LVU221
	.uleb128 0
.LLST46:
	.byte	0x4
	.uleb128 .LVL53-.Ltext0
	.uleb128 .LVL54-1-.Ltext0
	.uleb128 0x1
	.byte	0x51
	.byte	0x4
	.uleb128 .LVL54-1-.Ltext0
	.uleb128 .LFE5-.Ltext0
	.uleb128 0x4
	.byte	0xa3
	.uleb128 0x1
	.byte	0x51
	.byte	0x9f
	.byte	0
.LVUS2:
	.uleb128 0
	.uleb128 .LVU36
	.uleb128 .LVU36
	.uleb128 0
.LLST2:
	.byte	0x4
	.uleb128 .LVL6-.Ltext0
	.uleb128 .LVL9-.Ltext0
	.uleb128 0x1
	.byte	0x55
	.byte	0x4
	.uleb128 .LVL9-.Ltext0
	.uleb128 .LFE2-.Ltext0
	.uleb128 0x1
	.byte	0x58
	.byte	0
.LVUS3:
	.uleb128 .LVU26
	.uleb128 .LVU37
	.uleb128 .LVU37
	.uleb128 0
.LLST3:
	.byte	0x4
	.uleb128 .LVL7-.Ltext0
	.uleb128 .LVL10-.Ltext0
	.uleb128 0x2
	.byte	0x30
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL10-.Ltext0
	.uleb128 .LFE2-.Ltext0
	.uleb128 0x1
	.byte	0x55
	.byte	0
.LVUS4:
	.uleb128 .LVU27
	.uleb128 .LVU37
	.uleb128 .LVU37
	.uleb128 0
.LLST4:
	.byte	0x4
	.uleb128 .LVL7-.Ltext0
	.uleb128 .LVL10-.Ltext0
	.uleb128 0x2
	.byte	0x30
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL10-.Ltext0
	.uleb128 .LFE2-.Ltext0
	.uleb128 0x1
	.byte	0x54
	.byte	0
.LVUS6:
	.uleb128 .LVU29
	.uleb128 .LVU67
	.uleb128 .LVU97
	.uleb128 0
.LLST6:
	.byte	0x4
	.uleb128 .LVL7-.Ltext0
	.uleb128 .LVL18-.Ltext0
	.uleb128 0x2
	.byte	0x30
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL26-.Ltext0
	.uleb128 .LFE2-.Ltext0
	.uleb128 0x2
	.byte	0x32
	.byte	0x9f
	.byte	0
.LVUS8:
	.uleb128 .LVU30
	.uleb128 .LVU37
	.uleb128 .LVU37
	.uleb128 .LVU54
	.uleb128 .LVU54
	.uleb128 .LVU64
	.uleb128 .LVU64
	.uleb128 .LVU67
	.uleb128 .LVU67
	.uleb128 .LVU84
	.uleb128 .LVU84
	.uleb128 .LVU94
	.uleb128 .LVU94
	.uleb128 0
.LLST8:
	.byte	0x4
	.uleb128 .LVL8-.Ltext0
	.uleb128 .LVL10-.Ltext0
	.uleb128 0x2
	.byte	0x30
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL10-.Ltext0
	.uleb128 .LVL14-.Ltext0
	.uleb128 0x1
	.byte	0x52
	.byte	0x4
	.uleb128 .LVL14-.Ltext0
	.uleb128 .LVL17-.Ltext0
	.uleb128 0x3
	.byte	0x72
	.sleb128 -1
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL17-.Ltext0
	.uleb128 .LVL18-.Ltext0
	.uleb128 0x1
	.byte	0x52
	.byte	0x4
	.uleb128 .LVL18-.Ltext0
	.uleb128 .LVL22-.Ltext0
	.uleb128 0x1
	.byte	0x5a
	.byte	0x4
	.uleb128 .LVL22-.Ltext0
	.uleb128 .LVL25-.Ltext0
	.uleb128 0x3
	.byte	0x7a
	.sleb128 -1
	.byte	0x9f
	.byte	0x4
	.uleb128 .LVL25-.Ltext0
	.uleb128 .LFE2-.Ltext0
	.uleb128 0x1
	.byte	0x5a
	.byte	0
.LVUS10:
	.uleb128 .LVU46
	.uleb128 .LVU62
	.uleb128 .LVU76
	.uleb128 .LVU92
.LLST10:
	.byte	0x4
	.uleb128 .LVL13-.Ltext0
	.uleb128 .LVL17-.Ltext0
	.uleb128 0x1
	.byte	0x58
	.byte	0x4
	.uleb128 .LVL21-.Ltext0
	.uleb128 .LVL25-.Ltext0
	.uleb128 0x1
	.byte	0x58
	.byte	0
.LVUS11:
	.uleb128 .LVU48
	.uleb128 .LVU57
	.uleb128 .LVU57
	.uleb128 .LVU60
	.uleb128 .LVU78
	.uleb128 .LVU87
	.uleb128 .LVU87
	.uleb128 .LVU90
.LLST11:
	.byte	0x4
	.uleb128 .LVL13-.Ltext0
	.uleb128 .LVL15-.Ltext0
	.uleb128 0x1
	.byte	0x5a
	.byte	0x4
	.uleb128 .LVL15-.Ltext0
	.uleb128 .LVL16-.Ltext0
	.uleb128 0x1
	.byte	0x51
	.byte	0x4
	.uleb128 .LVL21-.Ltext0
	.uleb128 .LVL23-.Ltext0
	.uleb128 0x1
	.byte	0x59
	.byte	0x4
	.uleb128 .LVL23-.Ltext0
	.uleb128 .LVL24-.Ltext0
	.uleb128 0x1
	.byte	0x52
	.byte	0
.LVUS12:
	.uleb128 .LVU49
	.uleb128 .LVU62
	.uleb128 .LVU79
	.uleb128 .LVU92
.LLST12:
	.byte	0x4
	.uleb128 .LVL13-.Ltext0
	.uleb128 .LVL17-.Ltext0
	.uleb128 0x1
	.byte	0x59
	.byte	0x4
	.uleb128 .LVL21-.Ltext0
	.uleb128 .LVL25-.Ltext0
	.uleb128 0x1
	.byte	0x50
	.byte	0
.LVUS0:
	.uleb128 .LVU4
	.uleb128 .LVU12
	.uleb128 .LVU12
	.uleb128 .LVU16
	.uleb128 .LVU16
	.uleb128 .LVU20
.LLST0:
	.byte	0x4
	.uleb128 .LVL1-.Ltext0
	.uleb128 .LVL3-.Ltext0
	.uleb128 0x1
	.byte	0x50
	.byte	0x4
	.uleb128 .LVL3-.Ltext0
	.uleb128 .LVL4-.Ltext0
	.uleb128 0x1
	.byte	0x51
	.byte	0x4
	.uleb128 .LVL4-.Ltext0
	.uleb128 .LVL5-.Ltext0
	.uleb128 0xa
	.byte	0x75
	.sleb128 0
	.byte	0x6
	.byte	0x47
	.byte	0x24
	.byte	0x75
	.sleb128 0
	.byte	0x6
	.byte	0x27
	.byte	0x9f
	.byte	0
.LVUS1:
	.uleb128 .LVU6
	.uleb128 0
.LLST1:
	.byte	0x4
	.uleb128 .LVL2-.Ltext0
	.uleb128 .LFE1-.Ltext0
	.uleb128 0x1
	.byte	0x52
	.byte	0
.Ldebug_loc3:
	.section	.debug_aranges,"",@progbits
	.long	0x2c
	.value	0x2
	.long	.Ldebug_info0
	.byte	0x8
	.byte	0
	.value	0
	.value	0
	.quad	.Ltext0
	.quad	.Letext0-.Ltext0
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
.LLRL5:
	.byte	0x4
	.uleb128 .LBB20-.Ltext0
	.uleb128 .LBE20-.Ltext0
	.byte	0x4
	.uleb128 .LBB36-.Ltext0
	.uleb128 .LBE36-.Ltext0
	.byte	0x4
	.uleb128 .LBB37-.Ltext0
	.uleb128 .LBE37-.Ltext0
	.byte	0
.LLRL7:
	.byte	0x4
	.uleb128 .LBB21-.Ltext0
	.uleb128 .LBE21-.Ltext0
	.byte	0x4
	.uleb128 .LBB34-.Ltext0
	.uleb128 .LBE34-.Ltext0
	.byte	0x4
	.uleb128 .LBB35-.Ltext0
	.uleb128 .LBE35-.Ltext0
	.byte	0
.LLRL9:
	.byte	0x4
	.uleb128 .LBB22-.Ltext0
	.uleb128 .LBE22-.Ltext0
	.byte	0x4
	.uleb128 .LBB29-.Ltext0
	.uleb128 .LBE29-.Ltext0
	.byte	0x4
	.uleb128 .LBB30-.Ltext0
	.uleb128 .LBE30-.Ltext0
	.byte	0x4
	.uleb128 .LBB31-.Ltext0
	.uleb128 .LBE31-.Ltext0
	.byte	0x4
	.uleb128 .LBB32-.Ltext0
	.uleb128 .LBE32-.Ltext0
	.byte	0x4
	.uleb128 .LBB33-.Ltext0
	.uleb128 .LBE33-.Ltext0
	.byte	0
.LLRL24:
	.byte	0x4
	.uleb128 .LBB39-.Ltext0
	.uleb128 .LBE39-.Ltext0
	.byte	0x4
	.uleb128 .LBB56-.Ltext0
	.uleb128 .LBE56-.Ltext0
	.byte	0x4
	.uleb128 .LBB57-.Ltext0
	.uleb128 .LBE57-.Ltext0
	.byte	0x4
	.uleb128 .LBB58-.Ltext0
	.uleb128 .LBE58-.Ltext0
	.byte	0x4
	.uleb128 .LBB59-.Ltext0
	.uleb128 .LBE59-.Ltext0
	.byte	0x4
	.uleb128 .LBB60-.Ltext0
	.uleb128 .LBE60-.Ltext0
	.byte	0
.LLRL31:
	.byte	0x4
	.uleb128 .LBB41-.Ltext0
	.uleb128 .LBE41-.Ltext0
	.byte	0x4
	.uleb128 .LBB47-.Ltext0
	.uleb128 .LBE47-.Ltext0
	.byte	0x4
	.uleb128 .LBB48-.Ltext0
	.uleb128 .LBE48-.Ltext0
	.byte	0x4
	.uleb128 .LBB49-.Ltext0
	.uleb128 .LBE49-.Ltext0
	.byte	0x4
	.uleb128 .LBB50-.Ltext0
	.uleb128 .LBE50-.Ltext0
	.byte	0
.LLRL37:
	.byte	0x4
	.uleb128 .LBB62-.Ltext0
	.uleb128 .LBE62-.Ltext0
	.byte	0x4
	.uleb128 .LBB74-.Ltext0
	.uleb128 .LBE74-.Ltext0
	.byte	0x4
	.uleb128 .LBB75-.Ltext0
	.uleb128 .LBE75-.Ltext0
	.byte	0x4
	.uleb128 .LBB76-.Ltext0
	.uleb128 .LBE76-.Ltext0
	.byte	0x4
	.uleb128 .LBB77-.Ltext0
	.uleb128 .LBE77-.Ltext0
	.byte	0
.LLRL40:
	.byte	0x4
	.uleb128 .LBB64-.Ltext0
	.uleb128 .LBE64-.Ltext0
	.byte	0x4
	.uleb128 .LBB68-.Ltext0
	.uleb128 .LBE68-.Ltext0
	.byte	0x4
	.uleb128 .LBB69-.Ltext0
	.uleb128 .LBE69-.Ltext0
	.byte	0
.Ldebug_ranges3:
	.section	.debug_line,"",@progbits
.Ldebug_line0:
	.section	.debug_str,"MS",@progbits,1
.LASF31:
	.string	"bound1"
.LASF32:
	.string	"bound2"
.LASF13:
	.string	"uint64_t"
.LASF7:
	.string	"short int"
.LASF35:
	.string	"xorshift128plus_jump"
.LASF9:
	.string	"__uint32_t"
.LASF30:
	.string	"rand"
.LASF29:
	.string	"nextpos"
.LASF8:
	.string	"long int"
.LASF38:
	.string	"xorshift128plus_bounded"
.LASF28:
	.string	"bound"
.LASF14:
	.string	"part1"
.LASF15:
	.string	"part2"
.LASF33:
	.string	"bounded1"
.LASF34:
	.string	"bounded2"
.LASF2:
	.string	"unsigned char"
.LASF26:
	.string	"xorshift128plus_shuffle32_partial"
.LASF6:
	.string	"signed char"
.LASF12:
	.string	"uint32_t"
.LASF4:
	.string	"unsigned int"
.LASF23:
	.string	"val1"
.LASF25:
	.string	"val2"
.LASF3:
	.string	"short unsigned int"
.LASF19:
	.string	"lower_index_inclusive"
.LASF11:
	.string	"char"
.LASF27:
	.string	"xorshift128plus_shuffle32"
.LASF10:
	.string	"__uint64_t"
.LASF20:
	.string	"nextpos1"
.LASF21:
	.string	"nextpos2"
.LASF24:
	.string	"tmp2"
.LASF17:
	.string	"storage"
.LASF22:
	.string	"tmp1"
.LASF36:
	.string	"GNU C17 11.3.0 -mavx2 -mtune=generic -march=x86-64 -g -O3 -fasynchronous-unwind-tables -fstack-protector-strong -fstack-clash-protection -fcf-protection"
.LASF5:
	.string	"long unsigned int"
.LASF18:
	.string	"size"
.LASF41:
	.string	"xorshift128plus"
.LASF37:
	.string	"xorshift128plus_key_s"
.LASF16:
	.string	"xorshift128plus_key_t"
.LASF40:
	.string	"JUMP"
.LASF39:
	.string	"xorshift128plus_bounded_two_by_two"
	.section	.debug_line_str,"MS",@progbits,1
.LASF1:
	.string	"/home/sam/Desktop/2022-Fall-Parallel-Programming/HW2/part1"
.LASF0:
	.string	"SIMDxorshift/src//xorshift128plus.c"
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
