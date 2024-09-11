- [Characterization of regional differences in resting-state fMRI with a data-driven network model of brain dynamics | Science Advances](https://www.science.org/doi/10.1126/sciadv.abq7547)
- #[[Viktor Jirsa]]
- ![sciadv.abq7547.pdf](../assets/sciadv.abq7547_1726021778082_0.pdf)
-
- ## Abstract
	- 基于模型的全脑动力学数据分析将观察到的数据与神经群体网络中的模型参数联系起来。最近，研究重点是模型参数的区域方差的作用。然而，此类分析必然取决于预选神经群体模型的属性。我们引入了一种方法，可以**根据功能数据推断代表区域动态的神经群体模型以及区域和被试特定的参数**，同时尊重已知的网络结构。我们将该方法应用于人体静息态功能磁共振成像。我们发现**潜在的动态可以描述为围绕单个固定点的噪声波动**。该方法可靠地发现了三个在动力学中具有清晰和独特作用的区域参数，其中一个与基因表达空间图的第一主成分密切相关。本方法为静息态功能磁共振成像的分析开辟了一种新方法，并可能应用于理解衰老或神经退行性疾病期间的大脑动态。
-
- ## INTRODUCTION
	- 分析静息态功能磁共振成像 （fMRI） 的一种途径是使用大规模脑网络动力学的计算模型。这种方法的总体目标是**将观察到的大脑活动与计算模型的动力学库相关联**，可能通过识别最佳模型参数，从而更好地对观察结果进行机械解释。这些计算模型中有一类是基于网络的模型，其中节点代表大脑区域，边代表它们之间的结构连接。这些模型可以受到单个脑成像数据的限制;通常，扩散加权成像数据用于估计边缘权重。大脑区域的局部动力学由所谓的神经群体模型表示：神经元群体活动的低维模型。
	- 在将模型与观测结果联系起来时，直到最近，由于与探索高维参数空间相关的计算成本，研究只关注少量参数。通常，这些参数会全局影响网络动态，例如缩放网络中所有连接强度的参数。然而，近年来，一些工作使用全脑建模框架来探索模型参数的空间异质性的作用。具体来说，研究发现，当区域变异性受到 MRI 衍生的皮质内髓鞘含量估计、功能梯度或基因表达谱的限制时，全脑模型可以更好地再现静息态 fMRI 的特征，并且即使没有先验的限制，也发现了类似的区域变异性。
	- 这些研究中使用的神经群体模型 [例如基于电导的脉冲神经网络的动态平均场模型或神经兴奋性的 Hopf 分叉模型] 通常是通过一系列主要简化得出的，以得出简单的低维神经动力学模型。因此，可以质疑这些模型中包含的动力学结构在多大程度上足以捕捉观察到的数据中表现出的神经动力学的基本要素。**两个不同的神经质量模型会导致相同的结论，还是结果在很大程度上取决于确切的模型形式？**这些问题尚未得到充分回答。
	- **动态系统识别（dynamical system identification）**的最新进展为这个方向开辟了新的可能性，并提出了一个问题，即数据驱动的方法是否可以应用于全脑建模的背景下：我们能否学习一个代表大规模大脑网络每个节点的神经群体的动态系统？这种方法将使我们能够**回避对特定神经群体模型的依赖问题**，这是大规模建模的核心，而是**直接从功能数据中提取该模型**。
	- 我们这项工作的目标是使用基于网络的模型框架来学习全脑动力学模型（图 1）。也就是说，我们假设被连接的神经群体的模型形式，其连接强度来自弥散加权成像。然而，我们没有指定神经群体的功能形式，而是希望**从静息态 fMRI 中观察到的动力学中以数据驱动的方式学习神经群体模型**。使用这个模型，我们想**了解它在不同脑区和受试者之间变化的参数**，这与单个连接组网络一起，导致了跨脑区和受试者的观察到的动态差异。
	- **Fig. 1 Conceptual overview of the method.**
		- ![](https://www.science.org/cms/10.1126/sciadv.abq7547/asset/5ba3b762-8ef5-4606-a08e-ed2459b75640/assets/images/large/sciadv.abq7547-f1.jpg)
		- 该方法允许我们对大脑动力学的网络模型进行参数推断，其中每个节点（或大脑区域）的动力学模型最初是未知的。作为输入（左），它期望多个受试者队列的结构连接矩阵 $$w$$ 以及大脑活动 $$y$$ 的相应观察结果（例如脑区分割的静息态 fMRI）。受区域动力学通用模型结构（中）的约束，**它学习节点动力学f的动力学模型和状态到观测的投影模型g**。动力学模型 f 为所有受试者和区域共享，但它**取决于受试者特定参数** $$\theta_s$$ **和区域特定参数** $$\theta_r$$ 。这些也是从数据以及系统 $$x$$ 的隐藏状态和特定主题的外部输入 $$u_{ext}$$ 中推断出来的，由给定主题的所有区域共享。**所有系统状态、特定于主题和区域的参数以及外部输入都被概率地推断为正态分布，也就是说，我们推断每个参数的均值和方差。**
	- 鉴于这种方法避免了对区域动态特定形式的假设，所需的框架将有可能**独立验证传统方法中使用的方法与预定义的神经群体模型**（例如对区域动态性质的假设）并支持他们的发现（例如关于区域异质性在全脑动力学中的作用）。最终，这种方法可能会导致新一代更具表现力、数据驱动的大规模大脑动力学网络模型。
	- 这项工作的核心是**数据驱动的非线性动力系统识别问题**。近年来引入了许多方法，并应用于物理和生命科学的各个领域 （ 9 – 14 ），包括所有尺度的神经科学 （ 15 – 17 ）。动力学系统的表示和推理方法使用了多种方法：Brunton 等人。 （ 9 ） 引入了**非线性动力学的稀疏识别 （SINDy） 方法**，该方法使用稀疏回归来查找预定义的非线性函数库的系数，从而生成人类可读形式的常微分方程 （ODE）。一些作者通过**混合局部线性动力学来近似非线性系统**;这些方法包括用消息传递算法训练的**切换线性动态系统** （ 10 ， 13 ） 或变分推理学习固定点位置和相关的雅可比矩阵的方法 （ 11 ）。其他方法包括使用**期望最大化算法** （ 14 ， 16 ） 训练的**分段线性激活的递归神经网络**，依赖于**变分自动编码器**架构，用递归神经网络 （ 15 ） 表示动态系统，以及**非线性系统的分层参数化** （ 12 ）。这些方法在计算成本、结果重建的质量、实现的复杂性和结果系统的可解释性方面有所不同。
	- 我们的问题有几个特殊之处，无法直接采用现有的方法，因此我们开发了本文介绍的方法：
		- 1)我们希望在基于网络的大规模脑动力学模型的框架内操作，这意味着我们希望将已知的结构连通性纳入全脑模型。因此，该方法必须**允许预先指定的网络连接**。
		- 2)我们感兴趣的是噪声驱动的系统，即**系统由随机微分方程而不是确定性的微分方程来描述**。现有的一些方法只关注确定性微分方程，只考虑观测噪声，而不考虑底层系统中的噪声。
		- 3)它是神经群体的理想参量化，具有区域和被试参数。当遇到多个相关系统的系统辨识问题时，许多现有的方法并没有提供一种共享所学习动力学的方法，相反，这些系统是独立推断出来的。在这里，我们**只希望推断出一个动力系统，其被试间和区域间的差异仅通过区域和被试参数表示**。
		- 4)这是一个**部分观测问题**。有时假设所有的状态变量都被观察到，这样就大大简化了推理问题。我们假设我们**只有区域动力学的一维(1D)观察(如区域 fMRI 时间序列)** ，这意味着多个系统状态是隐藏的，也需要推断。
	- 为了解决这个问题，我们使用**摊销变分推理或变分自动编码器**（18）的框架，特别是受到其在推断神经群体动态（[Inferring single-trial neural population dynamics using sequential auto-encoders | Nature Methods](https://www.nature.com/articles/s41592-018-0109-9)）和具有层次结构的动态系统（[Efficient Amortised Bayesian Inference for Hierarchical and Nonlinear Dynamical Systems (mlr.press)](https://proceedings.mlr.press/v97/meeds19a.html)）中的应用的启发。简而言之，我们的系统由一个编码网络（**将观测到的时间序列映射到特定于被试和区域的参数以及源空间中的轨迹**）、表示动力系统的神经网络以及充当从源空间到观察空间解码器的观测模型组成。这些经过联合训练，以最大化**证据下限 (ELBO)**（[The evidence lower bound (ELBO) - Matthew N. Bernstein (mbernste.github.io)](https://mbernste.github.io/posts/elbo/)），以便训练模型的预测与原始数据非常相似。
	- 在这项工作中，我们在**两个合成数据集上测试**我们的方法，这些数据集是用大规模大脑建模中常用的两种模型生成的：基于电导的尖峰神经网络的平均场模型，或简称为平均场模型（7），以及Hopf 分岔模型 (8)。对于这两个测试用例，我们使用了由八名受试者组成的队列，这些受试者具有真实的结构连接组，并且模型参数在受试者和大脑区域之间有所不同。我们表明，经过训练的生成模型可以重现原始数据集的许多特征，并**证明该方法可以提取与用于模拟的原始参数密切相关的区域和特定于被试的参数**。
	- 最后，我们将该方法**应用于人类连接组计划 (HCP) 的 100 名受试者的静息态 fMRI 数据** (19)。我们发现，推断的动态可以描述为在节点和网络级别上围绕单个固定点的噪声波动。该方法可靠地发现了三个对动力学具有清晰和独特功能的区域参数，其中一个与基因表达空间图的第一主成分密切相关。我们进一步发现，**功能连接（FC）仅部分再现**，并且**严重依赖于所有区域共享的外部输入**，并且**仅在很小程度上依赖于网络交互**。
-
- ## RESULTS
	- 结果部分由三部分组成。在第一部分中，我们介绍了所开发方法的基本思想，它的详细描述可以在方法中找到。第二部分对合成数据，即参数已知的计算模型产生的数据进行了验证。第三部分将该方法应用于人体静息状态 fMRI 数据，并对结果进行分析。
	- ### Amortized variational inference for networks of nonlinear dynamical systems
		- 我们遵循大规模脑网络建模的总体框架，并假设对于特定被试，大脑区域 j 的观测值 $y_j(t))$ 由下述动力系统生成
			- $$\dot{X}_j(t) = f[X_j(t),\theta_j^r,\theta^s,u_{ext}(t),u_j(t)]+\eta_j(t)$$
			- $$y_j(t)=g[X_j(t)]+\nu_j(t)$$
		- 其中$$X_j(t)\in \mathbb{R}^{n_s}$$是t时刻的状态，$$\theta_j^r\in \mathbb{R}^{m_r}$$是区域特异的参数，$$\theta^s\in \mathbb{R}^{m_s}$$是被试特异的参数。
		- $u_{ext}(t)$是外部输入，由同一个被试的所有脑区共享。
		- $$u_j(t)=\sum_{i=1}^n{w_{ij}g_c[X_i(t)]}$$是区域j的网络输入，其中$w_{ij}$即结构连接矩阵。
		- 函数$f,g,g_c$均是未知的，$\eta_j(t)$和$\nu_j(t)$分别是系统噪声和观测噪声。
		- 从多个受试者的观察时间序列中，我们希望推断出对于所有受试者都相同的演化函数 f 和观察函数 g，以及特定于区域和受试者的参数 $$\theta_j^r$$ 和 $$\theta^s$$ 以及与时间相关的外部输入 $$u_{ext}$$。为此，我们采用摊销变分推理（18）的一般框架，其参数具有层次结构（图S1B）（12）。我们考虑状态 $$X_j$$，参数 $$\theta_j^r$$ 和 $$\theta^s$$，外部输入 $$u_{ext}$$ 作为潜在变量并寻求由多元高斯分布表示的近似后验分布。本着摊余变分推理的精神，我们不直接优化它们的参数，而是通过编码器函数 $$h_1$$、$$h_2$$、$$h_3$$ 和 $$h_4$$，将数据转换为潜在变量（分别为系统状态、区域和被试参数以及外部输入）。
		- 出于计算易处理性的原因，我们强假设观测函数和耦合函数是相同的，$g == g_c$。这使我们能够==有效地将网络问题解耦为已知网络输入的非耦合区域==，因此我们可以将一个被试的一个区域的时间序列视为单个数据点。我们根据讨论的结果回到这个选择及其可能的含义。
		- 我们用**通用人工神经网络表示非线性函数 f，将函数 g 表示为线性变换**。推理问题最终转化为成本函数 ELBO 的优化，**ELBO将在** f、g、$$h_1$$、$$h_2$$、$$h_3$$ 和 $$h_4$$ **的权重以及系统和观测噪声的方差上最大化**。优化后，我们获得了用函数 f 和 g 来描述动力系统，区域和被试参数 $$\theta_j^r$$ 和 $$\theta^s$$ 的概率表示以及外部输入 $$u_{ext}$$ 的值，以及状态空间 xj 中观测值的投影。推断的参数 $$\theta_j^r$$ 和 $$\theta^s$$ 不具有机制意义；然而，它们可以提供区域和被试的（不）相似性的度量，并且可以通过推断的动力系统 f 进行解释。
		- ### **关于变分推断**
		  collapsed:: true
			- #### amortized variational inference 是什么
			  collapsed:: true
				- [Amortized Variational Inference (AVI) 是一种用于近似概率模型中难以处理的后验分布的方法](https://arxiv.org/abs/2307.11018)[1](https://arxiv.org/abs/2307.11018)[。AVI 的主要特点是它学习一个全局推断函数，该函数将每个观察值映射到其局部潜变量的近似后验](https://arxiv.org/abs/2307.11018)[1](https://arxiv.org/abs/2307.11018)。
				- [在传统的变分推断中，需要对每个数据点优化 ξ_i，计算成本高，特别是当该优化被嵌入一个全局参数的更新循环时](https://www.dazhuanlan.com/2020/03/01/5e5b28acefd34/)[2](https://www.dazhuanlan.com/2020/03/01/5e5b28acefd34/)[。而 Amortized Inference 的基本想法是使用一个强大的预测器来基于 x_i 的特征预测最优的 z_i，即 z_i=f(x_i)](https://www.dazhuanlan.com/2020/03/01/5e5b28acefd34/)[2](https://www.dazhuanlan.com/2020/03/01/5e5b28acefd34/)[。这样，局部变分参数被一个函数替代，并且这个函数的参数是对所有数据点共享的，也就是说推断是 amortized（平摊、分期——没有标准术语翻译）的，不是一下子出来的](https://www.dazhuanlan.com/2020/03/01/5e5b28acefd34/)[2](https://www.dazhuanlan.com/2020/03/01/5e5b28acefd34/)。
				- [总的来说，Amortized Variational Inference 是一种有效和可扩展的方法，因为它利用参数化函数来学习近似后验密度参数](https://arxiv.org/abs/2209.10888)[3](https://arxiv.org/abs/2209.10888)[。这使得 AVI 比许多基于采样的技术更快](https://arxiv.org/abs/2209.10888)[3](https://arxiv.org/abs/2209.10888)[。然而，传统的 VI 算法无法扩展到大型数据集，并且无法在不重新运行优化过程的情况下立即推断超出范围的数据点](https://arxiv.org/abs/2209.10888)[3](https://arxiv.org/abs/2209.10888)[。最近在该领域的发展，如随机、黑箱和 amortized-VI，已经帮助解决了这些问题](https://arxiv.org/abs/2209.10888)[3](https://arxiv.org/abs/2209.10888)。
			- #### 你能给我一个例子吗?
			  collapsed:: true
				- [当然可以。在深度生成模型中，Amortized Variational Inference (A-VI)被用作一种计算技巧，以加速对局部潜变量的推断](https://arxiv.org/abs/2307.11018)[1](https://arxiv.org/abs/2307.11018)[。例如，考虑一个深度生成模型，如变分自编码器（VAE）。在这种模型中，我们有一个编码器网络，它将观察数据映射到潜在变量的分布，以及一个解码器网络，它将这些潜在变量映射回数据空间](https://arxiv.org/abs/2307.11018)[1](https://arxiv.org/abs/2307.11018)。
				- [在训练过程中，我们希望学习编码器和解码器的参数，以便最大化观察数据的对数似然](https://arxiv.org/abs/2307.11018)[1](https://arxiv.org/abs/2307.11018)[。然而，由于后验分布是不可解的，我们需要使用变分推断来近似它](https://arxiv.org/abs/2307.11018)[1](https://arxiv.org/abs/2307.11018)[。在传统的变分推断中，我们会为每个数据点学习一个独立的近似后验分布](https://arxiv.org/abs/2307.11018)[1](https://arxiv.org/abs/2307.11018)[。这需要大量的计算，并且难以扩展到大型数据集](https://arxiv.org/abs/2307.11018)[1](https://arxiv.org/abs/2307.11018)。
				- [相反，在 Amortized Variational Inference 中，我们学习一个全局的推断函数（即编码器），它将每个观察值映射到其局部潜变量的近似后验](https://arxiv.org/abs/2307.11018)[1](https://arxiv.org/abs/2307.11018)[。这样，我们就可以一次性地为所有数据点学习近似后验分布](https://arxiv.org/abs/2307.11018)[1](https://arxiv.org/abs/2307.11018)[。这大大减少了计算需求，并使得方法可以扩展到大型数据集](https://arxiv.org/abs/2307.11018)[1](https://arxiv.org/abs/2307.11018)。
				- [总的来说，Amortized Variational Inference 提供了一种有效且可扩展的方法来进行变分推断，并且已经被广泛应用于深度生成模型的训练和推断](https://arxiv.org/abs/2209.10888)[2](https://arxiv.org/abs/2209.10888)[1](https://arxiv.org/abs/2307.11018)。
			- #### 什么是变分推断
			  collapsed:: true
				- [变分推断（Variational Inference，简称VI）是贝叶斯近似推断方法中的一大类方法，它将后验推断问题巧妙地转化为优化问题进行求解](https://zhuanlan.zhihu.com/p/88336614)[1](https://zhuanlan.zhihu.com/p/88336614)[2](https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/102735795)[。相比另一大类方法马尔可夫链蒙特卡洛方法（Markov Chain Monte Carlo, MCMC），VI 具有更好的收敛性和可扩展性（scalability），更适合求解大规模近似推断问题](https://zhuanlan.zhihu.com/p/88336614)[1](https://zhuanlan.zhihu.com/p/88336614)[2](https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/102735795)。
				  [在变分推断中，我们不是直接求后验概率p(z|x)，而是求与其KL divergence最小的优化问题q*(z) = argmin(q(z)∈Q) KL(q(z)||p(z|x))](https://zhuanlan.zhihu.com/p/49401976)[3](https://zhuanlan.zhihu.com/p/49401976)[。其中，KL(q(z)||p(z|x)) = E[logq(z)] - E[logp(z|x)]，期望均是对于分布q(z)上的期望](https://zhuanlan.zhihu.com/p/49401976)[3](https://zhuanlan.zhihu.com/p/49401976)[。我们可以将其进一步展开为KL(q(z)||p(z|x)) = E[logq(z)] - E[logp(z,x)] + logp(x)](https://zhuanlan.zhihu.com/p/49401976)[3](https://zhuanlan.zhihu.com/p/49401976)[。注意这里最后一项为logp(x)，我们又回到最初的边缘分布的问题难以求解](https://zhuanlan.zhihu.com/p/49401976)[3](https://zhuanlan.zhihu.com/p/49401976)。
				- [由于我们无法直接计算KL，所以我们改变优化目标为与KL前两项相关的量evidence lower bound（简称ELBO）：ELBO(q)= E[logp(z,x)] -E[logq(z)]](https://zhuanlan.zhihu.com/p/49401976)[3](https://zhuanlan.zhihu.com/p/49401976)[。可以看到，ELBO是负的KL再加上logp(x)，由于logp(x)相对于q(z)的梯度为零，所以极小化KL divergence的问题与极大化ELBO的优化问题是等价的](https://zhuanlan.zhihu.com/p/49401976)[3](https://zhuanlan.zhihu.com/p/49401976)[。因此，变分推断将推断问题转化为了求极值的优化问题](https://zhuanlan.zhihu.com/p/49401976)[3](https://zhuanlan.zhihu.com/p/49401976)。
				- [总的来说，变分推断是一种强大且灵活的工具，它可以用于处理许多复杂的统计模型，并且已经在机器学习和统计领域中得到了广泛的应用](https://zhuanlan.zhihu.com/p/88336614)[1](https://zhuanlan.zhihu.com/p/88336614)[2](https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/102735795)。
			- #### 什么是马尔科夫链蒙特卡洛方法
			  collapsed:: true
				- [马尔科夫链蒙特卡洛方法（Markov Chain Monte Carlo，简称MCMC）是一种使用马尔科夫链进行随机抽样以进行近似数值计算的方法](https://zh.wikipedia.org/wiki/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E9%93%BE%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B)[1](https://zh.wikipedia.org/wiki/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E9%93%BE%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B)[。MCMC的基本思想是构建一条马尔科夫链，使其平稳分布就是要进行抽样的分布](https://zh.wikipedia.org/wiki/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E9%93%BE%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B)[1](https://zh.wikipedia.org/wiki/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E9%93%BE%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B)[。首先基于该马尔科夫链进行随机游走，产生样本的序列，然后使用该平稳分布的样本进行近似的数值计算](https://zh.wikipedia.org/wiki/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E9%93%BE%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B)[1](https://zh.wikipedia.org/wiki/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E9%93%BE%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B)。
				- [MCMC方法可以应用于概率分布的估计、定积分的近似计算、最优化问题的近似求解等问题，特别是被应用于统计学习中概率模型的学习与推理，是重要的统计学习计算方法](https://zhuanlan.zhihu.com/p/250146007)[2](https://zhuanlan.zhihu.com/p/250146007)。
				- [例如，假设你有一个复杂的多维概率分布，你想从这个分布中抽取样本。直接抽取可能非常困难，但是你可以构建一个马尔科夫链，其平稳分布就是你感兴趣的分布。然后，你可以让这个马尔科夫链运行一段时间，生成一系列样本。这些样本将会越来越接近你感兴趣的分布。最后，你可以使用这些样本来估计分布的各种性质，比如它的均值、方差等](https://zhuanlan.zhihu.com/p/37121528)[3](https://zhuanlan.zhihu.com/p/37121528)[2](https://zhuanlan.zhihu.com/p/250146007)[1](https://zh.wikipedia.org/wiki/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E9%93%BE%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B)。
				- [总的来说，MCMC提供了一种强大而灵活的工具，可以用来处理许多复杂的统计问题](https://zhuanlan.zhihu.com/p/37121528)[3](https://zhuanlan.zhihu.com/p/37121528)[2](https://zhuanlan.zhihu.com/p/250146007)[1](https://zh.wikipedia.org/wiki/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E9%93%BE%E8%92%99%E7%89%B9%E5%8D%A1%E6%B4%9B)。
			- #### 什么是马尔科夫链
			  collapsed:: true
				- [马尔科夫链（Markov Chain）是一种随机过程，它具有“无记忆性”，即下一状态的概率分布只能由当前状态决定，在时间序列中它前面的事件均与之无关](https://zhuanlan.zhihu.com/p/448575579)[1](https://zhuanlan.zhihu.com/p/448575579)[2](https://zh.wikipedia.org/wiki/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E9%93%BE)[。这种特定类型的“无记忆性”称作马尔科夫性质](https://zhuanlan.zhihu.com/p/448575579)[1](https://zhuanlan.zhihu.com/p/448575579)[2](https://zh.wikipedia.org/wiki/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E9%93%BE)。
				- [马尔科夫链可以是离散状态空间（一组随机变量的可能值的集合）或者有离散索引集合（通常表示时间），因此，马尔科夫链有许多变种](https://zhuanlan.zhihu.com/p/448575579)[3](https://blog.csdn.net/qq_43428310/article/details/106684182)[。通常所说的「马尔科夫链」是指具有离散时间集合的过程，也就是离散时间马尔可夫链（DTMC）](https://zhuanlan.zhihu.com/p/448575579)[3](https://blog.csdn.net/qq_43428310/article/details/106684182)。
				- [马尔科夫链作为实际过程的统计模型具有许多应用。在马尔科夫链的每一步，系统根据概率分布，可以从一个状态变到另一个状态，也可以保持当前状态](https://zhuanlan.zhihu.com/p/448575579)[2](https://zh.wikipedia.org/wiki/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E9%93%BE)[。状态的改变叫做转移，与不同的状态改变相关的概率叫做转移概率](https://zhuanlan.zhihu.com/p/448575579)[2](https://zh.wikipedia.org/wiki/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E9%93%BE)[。例如，随机漫步就是马尔科夫链的一个例子。在随机漫步中每一步的状态是在图形中的点，每一步可以移动到任何一个相邻的点，在这里移动到每一个点的概率都是相同的（无论之前漫步路径是如何的）](https://zhuanlan.zhihu.com/p/448575579)[2](https://zh.wikipedia.org/wiki/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E9%93%BE)。
				- [总的来说，马尔科夫链是机器学习和人工智能的基石，在强化学习、自然语言处理、金融领域、天气预测、语音识别方面都有着极其广泛的应用](https://zhuanlan.zhihu.com/p/448575579)[1](https://zhuanlan.zhihu.com/p/448575579)。
			- 什么是 amortized variational inference
			  collapsed:: true
				- [Amortized Variational Inference : An Overview | by Sertis | Medium](https://sertiscorp.medium.com/amortized-variational-inference-an-overview-f246c1e11e11)
				- [[2307.11018] Amortized Variational Inference: When and Why? (arxiv.org)](https://arxiv.org/abs/2307.11018)
				- [[2209.10888] Amortized Variational Inference: A Systematic Review (arxiv.org)](https://arxiv.org/abs/2209.10888)
				- [Collapsed Amortized Variational Inference for Switching Nonlinear Dynamical Systems (mlr.press)](https://proceedings.mlr.press/v119/dong20e.html)
				- 变分推断基础：
					- 变分推断是一种用于近似复杂概率模型中后验分布的方法。它的目标是找到一个易于处理的分布来近似真实的后验分布。传统的变分推断方法为每个数据点单独优化变分参数。
				- 摊销变分推断的定义：
					- 摊销变分推断是变分推断的一种改进形式。它使用一个参数化的模型（通常是神经网络）来直接从数据预测变分参数，而不是为每个数据点单独优化这些参数。这种方法"摊销"了推断的计算成本，使得在新数据上的推断变得更快。
				- 神经网络的应用：
					- 在摊销变分推断中，深度学习模型被广泛用作随机函数，用于创建变分分布的参数。这种方法利用了神经网络强大的表达能力和泛化能力。
				- 优势：
					- 效率：一旦训练完成，对新数据点的推断变得非常快速。
					- 泛化：学习到的推断模型可以泛化到未见过的数据。
					- 可扩展性：适用于大规模数据集和在线学习场景。
				- 应用领域：
					- 摊销变分推断在多个机器学习领域都有应用，包括但不限于：
						- 变分自编码器（VAEs）
						- 主题模型
						- 时间序列分析
						- 强化学习
				- 挑战和研究方向：
					- 尽管摊销变分推断带来了许多优势，但它也面临一些挑战：
						- 模型选择：如何选择合适的神经网络架构来参数化推断模型。
						- 优化难度：训练过程可能比传统变分推断更加复杂。
						- 理论保证：需要更多的理论工作来理解其性能界限和收敛性质。
				- 具体应用示例：
					- 在非线性动态系统中，研究者提出了一种基于摊销变分推断的方法来处理切换非线性动态系统。这种方法能够有效地推断系统状态和模式切换，展示了摊销变分推断在复杂时间序列模型中的应用潜力。
				- 总结：
					- 摊销变分推断是一种创新的方法，它结合了深度学习的强大功能和变分推断的统计基础。通过学习一个通用的推断模型，它大大提高了推断的效率和可扩展性。这种方法在处理大规模数据集和复杂概率模型时特别有效，为机器学习中的概率推断开辟了新的可能性。随着研究的深入，我们可以期待看到这种方法在更多领域的应用和理论上的进一步发展。
	- ### Validation on synthetic data
		- #### *Evaluation workflow*
			- 我们在两个合成数据集上测试了所提出的方法，其中数据由全脑建模中常用的模型生成。首先是 **Hopf 分叉模型** （ 8 ），如图 2 所示。这是一个双方程神经群体模型，其中，根据分岔参数 ai 的值，动力学**要么是围绕稳定固定点驱动的噪声（对于ai< 0），要么是频率为 fi 的振荡（对于ai> 0）**。在合成数据集中，这两个参数在不同脑区随机变化。第二个模型是**参数平均场模型 （pMFM）** （ 7 ），如图 3 所示。这是一个单方程模型，根据网络输入，它可以被推入**高单稳态、低单稳态或双稳态**。状态之间的切换是由噪声驱动的，我们改变了不同大脑区域的噪声强度。
			- **Fig. 2****. Hopf model test case: example subject.**
				- ![](https://www.science.org/cms/10.1126/sciadv.abq7547/asset/f94fa33f-3120-40f1-908e-c0e44ed1b3f5/assets/images/large/sciadv.abq7547-f2.jpg)
				- (A) 训练数据是使用大脑动力学的网络模型进行模拟的，其中在每个节点上放置了一个Hopf神经质量模型。
				- (B) 这些节点通过源自扩散加权成像的连接组进行耦合，并由特定于受试者的耦合参数G进行缩放。
				- (C) 分岔参数ai和固有频率fi的值在不同脑区之间有所变化。
				- (D) 原始模型生成的时间序列，包括三个示例（底部）和计算得出的功能连接（FC）（右侧）。
				- (E) 所有区域的推断区域参数（左上；突出显示的示例节点）和推断的特定于受试者的参数（左下；灰色显示，与数据集中所有受试者的参数一起）。十字/线条的跨度对应于推断高斯分布的两个标准差。底部：由于标准差较小，添加了圆圈以便于视觉辅助。右侧是三个示例节点在状态空间中的推断动态。矢量场是在假设零网络输入和推断参数的情况下评估的。背景颜色表示速度幅度；黄线表示节点活动的示例模拟时间序列；黑面和白面圆圈分别表示稳定和不稳定的固定点。在最上面的面板中，一个不稳定固定点和一个稳定固定点在视觉上重叠。
				- (F) 推断的区域参数，颜色由分岔参数ai（上）和频率fi（下）的真实值决定。分岔参数与推断的θ相关，而频率仅在振荡区域（即ai > 0的区域）与θ相关。
				- (G) 使用训练后的模型和推断参数生成的时间序列的一个示例。数据的重要特征在区域水平（幅度和频率）和网络水平（功能连接）上都得到了保留。
			- **Fig. 3****. pMFM test case: example subject.**
				- ![](https://www.science.org/cms/10.1126/sciadv.abq7547/asset/ebb85a5e-3100-4555-a306-0612b24f36a4/assets/images/large/sciadv.abq7547-f3.jpg)
				- （A至C) 训练数据是使用pMFM神经质量网络进行模拟的。根据网络输入，这些神经群体可能被迫进入单稳态（下态或上态）或双稳态。动力学由噪声驱动，噪声强度在不同区域间有所变化。
				- (D) 使用原始模型生成的时间序列和功能连接（FC）。底部面板显示了三个示例，右侧是100秒的时间窗口。
				- (E) 推断的区域参数（左上）和特定于受试者的参数。由于标准差较小，添加了圆圈以便于视觉辅助。右侧是三个示例节点在状态空间中的推断动态。矢量场是在假设网络输入u = 1的情况下，使用推断的参数进行评估的。背景颜色表示速度幅度；黄线表示节点活动的示例模拟时间序列；黑面和白面圆圈分别表示稳定和不稳定的固定点。
				- (F) 推断的区域参数，颜色由噪声强度参数的真实值决定（上）。原始参数存储在推断参数空间的对角线上。下：根据原始时间序列的平均值进行着色，这不代表原始模型参数，而是一个数据特征。
				- (G) 使用训练后的模型和推断参数生成的时间序列的一个示例。区域特定的特征（在状态之间切换和噪声程度）得到了很好的保留。区域相关性的结构也得到了再现，但相关性比原始数据弱。请注意，即使区域时间序列表现出双稳态的迹象（上态和下态），**这种双稳态可能是在网络层面上产生的，而不一定是在区域层面上**，这一点从(E)中的相位图可以看出。
			- 这两个模型都用于生成 8 个被试的合成数据，每个被试都有单独的结构连接组，其中包含 68 个 Desikan-Killiany 分块的皮层区域 （20）。连接组由全局耦合强度 G 缩放，我们将其设置为 Hopf 模型的被试之间线性增加，pMFM模型中则是为每个被试设置到最优值。
			- 为了确定所描述方法的性能，我们进行如下操作。首先，我们使用原始模型和区域参数的随机值（图 2D 和 3D）模拟数据。接下来，使用八个受试者的整个数据集，我们训练模型，同时获得由动力学系统的函数 f 描述的训练后的生成模型，以及被试和区域特定参数的概率表示（图 2E 和 3E）。从参数的后验分布中随机采样，并使用随机系统和观察噪声，我们使用训练后的模型重复生成新的时间序列（图 2G 和 3G）。
			- 我们基于以下标准评估训练模型的质量：首先，我们确定**推断出的参数是否与模型的原始参数相关**（参见图2F、3F以及4A和E）；其次，我们希望评估**生成的时间序列特征是否与原始时间序列相似**，这包括区域层面（参见图4B和F）和网络层面（参见图4C、D、G和H）。特别指出，我们使用从后验预测分布中采样的时间序列来评估这种相似性。也就是说，这些时间序列**是使用推断出的参数θr和θs生成的**，但使用随机的系统噪声和观测噪声。**这与从方程26中的后验分布q(x | y, u, c)采样是不同的**。这种评估方法允许我们检验模型不仅能够拟合给定的数据，还能够生成与原始数据具有相似特征的新数据，从而验证模型的泛化能力和对底层动态的捕捉程度。
			- **Fig. 4****. Quantitative evaluation of the synthetic test cases.**
				- https://www.science.org/cms/10.1126/sciadv.abq7547/asset/7cb3fb05-1703-4c1c-aa78-494c217918ce/assets/images/large/sciadv.abq7547-f4.jpg
				- 上排：Hopf模型。下排：pMFM模型。
				- (A和E) 原始参数与推断的区域特定参数（Hopf模型的分岔参数a和频率f，pMFM的噪声强度）和受试者特定参数（耦合强度G）的最优投影之间的非线性相关性。
				- (B和F) 原始时间序列的区域特征与训练后模型生成的特征之间的拟合度。对于Hopf模型，我们展示了时间序列频谱的余弦相似度和方差的差异；对于pMFM，我们展示了观察空间分布的Wasserstein距离和切换次数对数的差异。这些评估基于图2和图3中的示例，所有由训练后模型生成的时间序列，以及正文中描述的替代数据。
				- (C和G) 原始时间序列与生成时间序列的功能连接（FC）之间的拟合度。
				- (D和H) FC矩阵非对角元素的平均值。对于两种模型，相关强度都被低估了，尽管结构得到了保留。在所有面板中，条形显示了（5，95）百分位区间，点表示中位数。统计数据是从8个受试者的后验分布的50个样本中计算得出的[在(A)、(B)、(E)和(F)中分组]，以及68个区域（对于区域特定参数和特征）。使用原始模型的替代分布的统计数据也是从50个样本中计算得出的。
			-
		- #### *Inferred parameters correspond to the original model parameters*
			- 图 2F 中的示例显示了原始区域参数如何与 Hopf 模型的推断参数 θr 相关联。分岔参数 a 映射到推断的参数 $$\theta^r_1$$（顶部），而频率 f 映射到 $$\theta^r_2$$（底部）。然而，后者仅适用于处于振荡状态的区域，即 a > 0。这并不是所提出的方法的缺点：在稳定点状态下，活动主要是噪声驱动的，频率参数的值的影响很小，甚至可以忽略不计（参见图 2D 中的示例 C）。换句话说，无法从数据中识别参数。这反映在推断的参数中。对于 a>0 的区域，推断的参数具有低方差，并且它们的平均值映射到原始频率参数。但是，对于 a < 0 的区域，推断的方差较高，接近前验值 1，并且分布重叠，这表明在此方案中存储的信息不多。
			- 此外，对于 pMFM 测试用例，噪声强度参数得到了很好的识别（图 3F）;但是，特定于区域的参数的第二个维度与区域时间序列的平均值相关。据推测，这是为了让参数可以补偿较弱的网络耦合，我们将在后面讨论。对于这两个示例，特定被试的耦合强度被映射到被试参数 θs（图 2E 和 3E，底部）
			- 拟合优度的定量分析如图 4 （A 和 E） 所示。为了评估它，对于每个原始参数，我们首先通过对后验分布均值进行线性回归来确定参数空间中存储参数的方向。然后，我们反复从参数的后验分布中取样，将它们投影到识别出的子空间上，并计算非线性 Spearman 相关系数 ρ。对于大多数参数，这些值都接近于最优值 1，这表明原始参数确实在推断的参数中被准确恢复。由于上面讨论的不可识别性，频率 f 是例外。但是，如果我们将区域仅限于分岔参数为正的区域，那么相关性就会显著增加，正如所讨论示例所预期的那样。参数相关性和报告的统计数据的进一步说明见图。S2 和 S3。
			- 在图 .S4 中，我们进一步评估了 Hopf 模型中拟合优度如何随着耦合的增加而变化。据推测，**随着耦合的增加，区域时间序列受连通区域活动的影响更大，受其内部参数的影响较小，因此从数据中恢复原始参数更加困难。**这就是我们在振荡状态下观察到的节点的分岔参数 a 和频率 f 的趋势。
		- #### *Trained model reproduces the features of regional time series*
			- 训练模型的关键测试是评估生成的数据是否与用于初始训练的数据相似。**这种相似性不应被理解为复制时间序列，因为它们依赖于特定的噪声实例化，而是应该保留我们认为有意义的特征**。对于这两个测试用例，我们评估了两个特征的相似性。对于具有振荡动力学的 Hopf 模型，我们评估了原始时间序列和生成**时间序列的光谱的余弦相似性以及时间序列方差之间的差异**，因为振荡和定点状态下的节点方差之间的差异很大（图 4B 和图 S5）。对于 pMFM，我们使用两个**分布的 Wasserstein 距离**（也称为Earth mover's distance, EMD），根据 1D 观测空间中的分布（即取随时间折叠的样本）来比较时间序列。pMFM 时间序列的第二个特征是 **up-down 状态之间开关的对数缩放次数**，捕捉了开关动态的时间方面（图 4F 和图 S6）。
			- 我们评估 50 个不同噪声实例的测量，导致每个区域有 50 个不同的时间序列，获得拟合优度指标的分布。还针对三个替代数据评估相同的指标：第一个是原始计算模型，使用不同的噪声实例运行。考虑到这些特征必然取决于初始模拟中使用的特定噪声实例，这提供了对拟合优度方面可以实现的乐观估计。第二个代理是通过在区域和受试者之间随机重新排列原始数据而获得的。第三个替代项只是白噪声，均值为零，方差等于一（由于数据归一化，它等于所有受试者和区域的原始数据集的均值和方差）。
			- 在大多数指标中，**训练后的模型的表现与原始模型相当或稍差**，但明显优于重新洗牌的替代模型和噪声替代模型（图 4、B 和 F；统计测试的数值和结果见表 S2）。例外是光谱与 Hopf 模型的余弦相似度。在这个衡量标准中，重新洗牌代理达到了与训练模型相似的值；这是因为耦合网络中的许多区域以相似的频率振荡。因此，两个随机选择区域的时间序列通常在其光谱的余弦相似性方面达到高分。
		- #### *Functional network structure is reproduced but with lowered strength*
			- 正如训练有素的模型应该能够在单个区域级别上重现原始数据的特征一样，它也应该能够在网络级别上重现相关特征。具体来说，我们评估 FC 的再现效果。一般来说，FC 量化了大脑区域时间序列之间的统计依赖性。虽然有多种方法来衡量它，但最普遍的是时间序列的线性（皮尔逊）相关性，我们在这里也使用它。这个静态 FC 捕捉统计相似性的空间结构；然而，它也有其局限性。值得注意的是，它忽略了 FC 结构（21、22）的时间变化。
		- #### *Large perturbations of the connectome lead to reduced performance*
			-
	- ### Application on human resting-state fMRI
		-
		- **Fig. 5****. Example of model inference on a single subject from the HCP dataset.**
			- https://www.science.org/cms/10.1126/sciadv.abq7547/asset/03a7acc6-5ace-4d57-aa9e-65257bd8c3f2/assets/images/large/sciadv.abq7547-f5.jpg
			-
		-
		- #### *Noisy dynamics around a single fixed point*
			-
			- **Fig. 6****. Discovered node and network dynamics.**
				- ![](https://www.science.org/cms/10.1126/sciadv.abq7547/asset/15957b06-036f-4027-ae48-3a08f4d366f9/assets/images/large/sciadv.abq7547-f6.jpg)
				-
			-
		- #### *Three region-specific parameters identified*
			-
			- **Table 1****. Quantification of the posterior collapse phenomenon for the regional variables θ****r**** as the dimensionality of the parameter space is increased.**
				- | $$m_r$$ | 1 | 2 | 3 | 4 | 5 |
				  | ---- | ---- | ---- |
				  | $$\theta^r_1$$ | 3.313 ± 0.268 | 2.946 ± 0.209 | 2.846 ± 0.135 | 2.744 ± 0.230 | 2.716 ± 0.133 |
				  | $$\theta^r_2$$ | – | 2.266 ± 0.935 | 2.025 ± 0.786 | 2.238 ± 0.754 | 2.048 ± 0.714 |
				  | $$\theta^r_3$$ | – | – | 0.917 ± 0.562 | 0.713 ± 0.601 | 0.881 ± 0.515 |
				  | $$\theta^r_4$$ | – | – | – | 0.002 ± 0.002 | 0.004 ± 0.007 |
				  | $$\theta^r_5$$ | – | – | – | – | 0.001 ± 0.000 |
			-
			- **Fig. 7****. Effect of the regional parameters on the generated data.**
				- ![](https://www.science.org/cms/10.1126/sciadv.abq7547/asset/69b3d55b-5dc8-4a8f-a7f0-2424c79bd896/assets/images/large/sciadv.abq7547-f7.jpg)
				-
			-
			- **Table 2****. Results of the multivariate linear regression between the means of the inferred regional parameters θ****r**** and regional features on the individual or population level.**
				- |   | Feature | *R*2 | Weights |
				  | ---- | ---- | ---- |
				  |   |   |   | $$\theta^r_1$$ | $$\theta^r_2$$ | $$\theta^r_3$$ |
				  | Individual data | SC: Node in-strength | 0.55* | −0.60* | 0.03† | −0.57* |
				  |  | SC: Node centrality | 0.35* | −0.41* | 0.17 | −0.48* |
				  |  | fMRI: First PCA eigenvector | 0.44* | −0.00† | 0.86* | 0.05 |
				  |  | fMRI: Second PCA eigenvector | 0.01 | 0.05 | 0.09 | 0.04† |
				  |  | fMRI: Correlation with mean signal | 0.58* | 0.10 | 1.00* | 0.15 |
				  |  | fMRI: Correlation with network input | 0.52* | −0.43* | 0.62* | 0.51* |
				  |  | fMRI: Number of zero-crossings | 0.96* | 0.99* | 0.02 | −0.01 |
				  |  | fMRI: Power below 0.1 Hz | 0.93* | −0.97* | −0.04 | 0.04 |
				  | External data | Neuronal size (Von Economo) | 0.17 | 0.38 | 0.20 | −0.14 |
				  |  | Neuronal density (Von Economo) | 0.25 | −0.46 | −0.22 | 0.20 |
				  |  | Neuronal density (BigBrain) | 0.19 | −0.42 | −0.24 | 0.02† |
				  |  | RSFC principal gradient | 0.07 | 0.27 | 0.07 | 0.03† |
				  |  | T1w/T2w ratio | 0.19 | −0.40 | −0.09 | 0.22 |
				  |  | Gene expression map (first PC) | 0.47* | −0.69* | −0.15 | 0.11 |
				  |  | EI map | 0.14 | −0.33 | −0.23 | −0.18 |
				-
			-
		- #### *Role of the network coupling in the trained model*
			-
			- **Fig. 8****. The role of the network connectivity and external input in reproducing the correlation structure.**
				- https://www.science.org/cms/10.1126/sciadv.abq7547/asset/91ddd14b-3b1b-45c0-8695-46b57a5791db/assets/images/large/sciadv.abq7547-f8.jpg
				-
			-
-
- ## DISCUSSION
	- ### Main results
		-
	- ### Discovered dynamics on the node and network level
		- #### *Optimization target*
			-
		- #### *Method of system identification*
			-
		- #### *fMRI preprocessing*
			-
		- #### *Fixed structural connectivity*
			-
		- #### *Coupling via the observed variable*
			-
		- #### *Related works on system identification from fMRI signals*
			-
		- #### *Cortical gradients in large-scale brain modeling*
			-
		- #### *Inferring subject-specific parameters*
			-
		- #### *Outlook on the role of dynamically relevant parameters*
			-
-
- ## METHODS
	- ### Structural connectomes
		-
	- ### Resting-state fMRI data
		-
	- ### Amortized variational inference for networks of nonlinear dynamical systems
		- #### *Generative dynamical system*
			-
		- #### *Evidence lower bound*
			-
	- ### Decoder or the observation model
		- #### *Prior on the system states*
			-
		- #### *Prior on the parameters*
			-
		- #### *Prior on the external input*
			-
		- #### *Approximate posteriors*
			-
	- ### Optimization
		-
	- ### Whole-brain network models for simulated datasets
		- #### *Hopf bifurcation model*
			-
		- #### *Parametric mean field model*
			-
	- ### Relation of regional parameters and regional features from individual and external data
		- #### *Features from individual data*
			-
		- #### *Features from external data*
			-
		- #### *Multivariate linear regression*
			-
	- ### Analysis of the inferred dynamical systems
		-