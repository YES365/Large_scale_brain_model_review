- [Whole-Brain Modelling: Past, Present, and Future | SpringerLink](https://link.springer.com/chapter/10.1007/978-3-030-89439-9_13)
- ## Abstract
	- 全脑建模是一个历史短暂但源远流长的科学领域。它的各种学科根源和概念成分早可以追溯到 20 世纪 40 年代。然而，直到 2000 年代末，一个新生的范式才以大致目前的形式出现——同时并在许多方面与其姊妹领域宏观连接组学紧密相连。这一时期出现了一些由著名理论和认知神经科学家组成的混合团队撰写的开创性论文，这些论文在很大程度上定义了 2020 年代初全脑建模的前景。与此同时，该领域在过去十年中已经扩展到十几个或更多令人着迷的新方法、理论和临床方向。在本章中，我们回顾了全脑建模的过去、现在和未来，指出了我们认为它最伟大的成功、最困难的挑战和最令人兴奋的机遇。
- # Whole-Brain Modelling: Past, Present, and Future
- ## The Past
	- “The transition point from The Past to The Present in our exposition is identified as the year 2006.”
- ### Origins of Neural Population Model
	- Neural field, 相比于 neural mass, mean field, 更加强调了空间信息。
	  
	  NPM的核心在于基于connectome连接的单个介观尺度的神经元集团的模型。之所以采取这样的方法在于我们的emperical data 也是介观的，由大量神经元核团产生M/EEG MRI 的神经信号。
	  
	  早期的NPM基于物理模型启发。McCulloch and Pitts (1943) 发现二值化的网络可以展现出一定的计算能力，奠定了System neuroscience 以及人工智能的基础。其后大量的人做了关于网络的统计方向的工作，其基础包括三个方面，“(1) a continuous time-dependent activity variable, (2) a linear weighted sum over these activity variables in the input, and (3) a nonlinear activation function linking input to output.” 其中包括了 WC 以及 JR的工作。顺着JR对EEG的一些特征信号的复现，David & Friston提出了DCM更好的拟合神经信号。
- ### Macroscopic Neural Field Model
	- 宏观的NFM一般将一个大脑两个半球整合起来近似成一个球体来研究。其中连接强度随距离指数下降。因此Nunez在1974年首次提出全脑的Brain Wave Equation。之后1997 Robinson 提出一个偏微分方程的BWE.
- ### Emergence of WBM(Whole Brain Model)
	- 从介观（小范围）的建模到全脑建模离不开Neuroimaging以及Network Science 的发展。
	  
	  Neuroimaging 的工作主要关注在FC，Functional Segregation, Functional Integration.
	  
	  Network Science 的工作包括 algebraic graph theory 以及 dynamical Systems. 
	  
	  “Particularly influential in the early years of network science was the concept of a ‘small-world network’, introduced by Watts and Strogatz”(1998).
	  
	  WBM的发展离不开Viktor Jirsa 2002 的一项工作，其主要发现为：“inhomogeneous and non-translationally invariant connectivity structure and time delays were essential for a faithful modelling of human largescale brain dynamics.”
- ## The Present
	- Three technical areas: (1) the canonical WBM equations, (2) physiological parameter estimation and inference, and (3) anatomical connectivity.
	- ### “The Canonical WBM Equations”
		- A generic or canonical formulation for connectome-based neural population models is given by stochastic delay integrodifferential equation.
		- #### Neural Population Model:
			- distinctions between Models could be ‘phenomenological’ and ‘physiological:
			  
			  “Physiological models attempt to describe, parsimoniously but accurately, relevant aspects of the physiology of neural tissue that determine dynamics at the population scale.”
			  
			  “Phenomenological models, in contrast, adopt the ‘spirit’ of the neural population activity idea”
		- #### Phenomenological Model:
		- #### Linear Diffusion(LD):
			- 非常简单的线性模型，经常被作为Null model 来对比
		- #### Kuramoto
			- 用单个state variable来表示node的活动，“this state variable is the phase of a limit cycle oscillator, giving a direct but quite abstract representation of rhythmic neural activity”
			  
			  常用Kuramoto来展现多个相位振子的synchronization dynamics. 这些振子最后通常会稳定在一个共振的频率。
			  
			  Kuramoto 引入了complex order parameter来宏观的展示全脑震荡的synchrony.
		- #### Stuart-Larndau(SL)
			- “two state nonlinear dynamical system that gives the canonical model (normal form) for an AndronovHopf bifurcation”
			  
			  近期的关于SL的研究主要围绕multisability, criticiality, metastability.
			  
			  “Metastability refers to the simultaneous realization of two competing tendencies to synchronize and desynchronize, achieved by spontaneous shifting between unstable states or transient attractor-like states (Scott Kelso, 2012)”
		- #### Fitzhugh-Nagumo(FHN)
			- 2 dimension reduction version of HHM
			  
			  “In WBM studies using FHN model, the natural frequency is usually set to either alpha or gamma.” (“Computational Modelling of the Brain: Modelling Approaches to Cells, Circuits and Networks”, 2022, p. 332)
		- #### Physiological  Models
			- 不同模型不同之处在于：“These differ along two primary dimensions: (a) the local circuit motif (number of neuronal subpopulations, and how they are connected), and (b) the dynamics—i.e. the equations used to describe activity in each subpopulation.”
		- #### Wilson-Cowan
			- “excitatory and inhibitory subpopulations, with coupled first-order nonlinear ordinary differential equations describing activity level”
		- #### Jansen-Rit
			- “features one inhibitory and two excitatory neural subpopulations, with second-order differential equations describing populationaverage membrane potential dynamics.”
		- #### Dynamic Mean Field
			- Wong-Wang Model
			  
			  Incorporate neurotransmitters
		- #### Robinson
			- “is a neural field model that includes a rhythm-generating corticothalamic circuit with nonzero conduction delay.”
	- ### Parameter Estimation
		- Tuning of Parameters
		  
		  Parameter Space Exploration-Brute force searching for optimal point in parameter space. Works best for 1-D and 2-D.
		  
		  Sampling Based approaches. 概率函数采样
		  
		  Evolutionary algorithms.
		  
		  Gradient-based approach: DCM
	- ### Connectivity
		- 这里讨论的是用于建模的脑区连接的data
		  
		  Tractography-based connectivity 会有 false positives and false negatives. 大量的FP来源于corssing fibers. FN 来源于一些长程的连接由于影像技术的复杂步骤被忽略掉。
		  
		  作者在这里讨论了一些Tractography的工作，对数据做出了许多调整，最后形成了我们现在看到的0-1的多脑区的DTI data.
		  
		  最后提到这些调整多种多样，导致了对模型的泛化能力和reproducibility的一些问题
	- ### Atlases
		- 讨论了CoCoMac以及 HCP dataset。
		  
		  Ghosh等将CoCoMac map 到人脑模型上
		  
		  Allen Mouse Brain Atlas
	- ### Clinics
		- WBM 对疾病贡献的几个方面：
		  
		  Mechanisms, Personlization, Virtual Therapies
		  
		  General Approach:模拟 病理学或药理学的干预1. 利用先验的病理学知识修改参数 2.对病例MRI拟合建模 3. 两个的结合
		- #### Epilepsy
			- 对癫痫的模型一般包括两种类型 1. bistability 2.bifurcation
			  
			  Taylor 2014 最早提出相关模型
			  
			  “Taylor et al. (2014) developed one of the first personalized connectome-based neural population models of epileptic activity.” (“Computational Modelling of the Brain: Modelling Approaches to Cells, Circuits and Networks”, 2022, p. 341)
			  
			  Jirsa 2014 的 epileptor 最富盛名 由五个变量描述 其中两个描述快速变化，另外两个描述wave event，最后一个描述正常state和癫痫state的转换。
			  
			  还有一些从graph theory 的视角来研究癫痫
		- #### Stroke & Neurodegeneration
			- “Falcon et al. (2015) conducted parameter space explorations with WBMs built from individual stroke patients’ connectivity matrices.”
			  
			  “Kaboodvand et al. (2019) also studied the effect of malfunction or lesions by changing each brain region’s local dynamics”
			  
			  And then **AD**:
			  
			  “One of the first WBM studies exploring this area was that of Zimmermann et al. (2018),”
			  
			  “Stefanovski et al. (2019) used the Jansen–Rit model to study changes in EEG in Alzheimer’s
		- #### Neuropsychiatry and Neuromodulation
			- Neuropsychiatry - schizophrenia: (proposal) “schizophrenia may constitute a form of ‘disconnection syndrome’”
			  
			  In a classic study, Cabral et al. (2011) investigated this using a WBM with Kuramoto dynamics.”and found disconnectivity in schizophrenia is considered to be relatively more global and diffuse.
			  
			  Others have explored the idea that functional disconnectivity patterns in schizophrenic brains may result from disrupted local dynamics, rather than anatomical connectivity disruptions per se. For example, using a WBM with two-state DMF dynamics, Yang et al. (2016) and Krystal et al. (2017).
			  
			  对精神类疾病的干预主要包括drugs & electrical and magnetic brain stimulation
			  
			  Many have done WBM based on serotonin type 2A receptor and spatial serotonergic connectivity.
			  
			  As for brain stimulation, there are rare WBM studies in a direct clinical context. A widespread adoption of brain stimulation is based on biophysical E-field model.
- ## The Future
	- ### Multiscale
		- Multiscale means using multiple models with different resolutions and complexity.
		  
		  Type A and Type B problem: 
		  
		  A: 微观模型用于衡量一些有空间特征的现象，而宏观模型用于衡量网络的连接 这类建模一般用于研究recorded macro neuroimaging mearsurement
		  
		  B: 连接仍然使用微观的信息
		  
		  “An example of a Type B problem would be to combine a WBM of ongoing activity with a biophysical model of electrical stimulation entering multiple brain regions.” “An interface is therefore needed to link the two levels of description, translating the physical, spatiotemporal electrical field pattern into a neural population activity input. In an excellent example of this approach (also noted above), Kunze et al. (2016)”
		  
		  还有两种相结合的研究：
		  
		  eg. “use Maxwell’s equations to calculate the E-field strengths, model stimulated areas with detailed neuron models, and describe the rest of the brain with neural populations.” The best model fit for this study could possibly be morphological neurons. Because the orientation and position of neuron could influence the magnetic and current field.
	- ### Standardized Model Construction
		- 模型的连接，heterogeneity现在依旧在发展，因此缺少一个统一的建模方法
		  
		  最常用MRI streamline 的问题在于“tractography algorithm outputs such as ‘streamline densities’ are highly artificial constructs that are not expressed in physically, physiologically, or anatomically meaningful quantities.”
		  
		  其他可能的方法：“‘connectivity models’ would be specified in biologically meaningful units, such as number or per-unit-area density of axons, and be constrained to exhibit certain statistical properties known from gold-standard tracer studies in non-human primates, such as exponential fall-off with distance and multiple orders of magnitude in weights.” Personalized model could be introduced by modulation to these basic model.
		  
		  另外可以加入一些subcortical region的连接
		  
		  Cross-Species
		  
		  动物实验的技术总是比人类的快，时间空间精度也更高，SNR也高