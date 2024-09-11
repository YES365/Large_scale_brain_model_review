- [Inversion of a large-scale circuit model reveals a cortical hierarchy in the dynamic resting human brain | Science Advances](https://www.science.org/doi/10.1126/sciadv.aat7854)
- #[[Thomas Yeo]]
- ## 简介：
	- 用大尺度环路模型反演皮层层次结构。
	- 使用rMFM（relaxed dynamical mean-field model），模型为脑区网络的动态演化模型，每个脑区的状态受4个变量影响：周期性连接/自反馈连接、其它脑区的输入、兴奋性皮质下输入、神经元噪声。
	- 使用结构数据构建脑区间的连接，使用高斯噪声模拟神经元噪声，使用功能数据来优化周期性连接和兴奋性皮质下输入。
	- 将优化得到的周期性连接和兴奋性皮质下输入与微观结构度量、功能连接梯度等做相关，发现这两个变量反映了从感觉运动皮层到默认活动网络的皮层层次结构。
-
- ## Abstract
	- 我们考虑了一个具有区域特异性微尺度特性的人类大脑皮层大规模动态电路模型。该模型使用随机优化方法进行了反向优化，得到了与新的、样本外静息功能性磁共振成像(fMRI)数据更加吻合的结果。在不假设层次结构存在的情况下，估计的模型参数揭示了大规模的皮层梯度。
	- 感觉运动区有更强的周期性连接和兴奋性皮层下输入，默认活动网络则具有更弱的周期性连接和兴奋性皮层下输入。
	- 周期性连接与其他宏观、微观尺度的皮层层次特征相关（认知功能元分析、静息态功能网络梯度、髓鞘化、 laminar-specific neuronal density）。
-
- ## Introduction
	- 存在微观和宏观的证据说明灵长类大脑具有层次结构，感觉区与联合区Association Cortex位于层次的两端：Distributed hierarchical processing in the primate cerebral cortex - PubMed  微观上，脑区间投射呈现层次特征（laminar pattern），解剖和影像手段都揭示了层次结构：Just a moment... ；Hierarchy of transcriptomic specialization across human cortex captured by structural neuroimaging topography | Nature Neuroscience 功能磁共振研究则提供了宏观证据：Is the rostro-caudal axis of the frontal lobe hierarchical? | Nature Reviews Neuroscience； Separate visual pathways for perception and action - PubMed；  Situating the default-mode network along a principal gradient of macroscale cortical organization | PNAS 但宏观证据与微观证据之间的关联性有待考证。
	- 脑区连接的大尺度生物物理模型仿真是沟通宏观微观的一种有效方式 。这类模型通过精简但有一定生物物理依据的神经质/场模型（parsimonious but biophysically plausible neural mass or neural field models  ）来模拟脑区活动。脑区活动模型组合上结构连接和血液动力学模型，可以试图再现静息态功能磁共振数据。模型基于微观机理但面向宏观数据，故而可以用于沟通微观和宏观。
	- 本文使用动态MFM模型（Resting-State Functional Connectivity Emerges from Structurally and Dynamically Shaped Slow Linear Fluctuations | Journal of Neuroscience）和DCM（dynamic causal modeling）对周期性连接和兴奋性皮质下输入进行优化。使用atlas of resting-state networks（The organization of the human cerebral cortex estimated by intrinsic functional connectivity | Journal of Neurophysiology）；认知功能元分析|meta-analysis（Functional Specialization and Flexibility in Human Association Cortex | Cerebral Cortex | Oxford Academic）；静息态功能连接第一主梯度（Situating the default-mode network along a principal gradient of macroscale cortical organization | PNAS）；皮层髓鞘化程度（A multi-modal parcellation of human cerebral cortex | Nature）；神经元大小/密度的估计（Bridging Cytoarchitectonics and Connectomics in Human Cerebral Cortex | Journal of Neuroscience ）来进行相关性分析。
	- 本文使用模型优化得到了层次结构，与使用与生物物理机制不直接相关的统计手段，或是简单假设参数服从解剖层次的研究形成对比（Just a moment...；Just a moment...）。
	- 周期性连接和兴奋性皮质下输入提供了互补的信息， 周期性连接强度在感觉运动和注意网络之间有所区别，皮质下输入区分了边缘系统、控制系统和默认系统，且只有前者与其他层次结构有显著相关。
-
- ## Results
	- ### Automatic optimization of rMFM parameters significantly improves agreement between simulated and empirical RSFC
		- 使用HCP S500数据集（The WU-Minn Human Connectome Project: An overview - ScienceDirect），Desikan-Killiany皮层分割|68脑区（An automated labeling system for subdividing the human cerebral cortex on MRI scans into gyral based regions of interest - ScienceDirect）分别对训练组和测试组计算组平均功能网络和结构网络。
		- 使用MFM模拟功能数据FC。MFM假设脑区的活动由4个变量驱动：自反馈连接、脑区间连接、兴奋性皮层下输入和神经元噪声。这对应4个可调节参数：w 为自反馈连接强度；G 为脑区间连接的尺度系数 ，脑区间连接由结构连接SC通过尺度系数放缩得到；I 为兴奋性皮层下输入，单位为nA；神经元噪声假设为高斯噪声，有参数\sigma 。
		- 此处模型认为w, I 是脑区间可变的， 共138个参数，称为rMFM。使用训练集做优化，而后在测试集上做模拟。
		- 以测试集上的FC矩阵的皮尔逊相关作为指标，rMFM的表现优于MFM，MFM模拟数据与实验数据的相关性仅和基线（SC与FC的相关性）相当。
		- **Fig. 1**** Automatic optimization of rMFM parameters yields stronger agreement between empirical and simulated RSFC.**
			- ![](https://www.science.org/cms/10.1126/sciadv.aat7854/asset/1f39865f-c6e1-4a48-84c5-7402db83ce1a/assets/graphic/aat7854-f1.jpeg)
		-
	- ### Sensory-motor systems exhibit strong recurrent connections and excitatory subcortical input, while the default network exhibits weak recurrent connections and excitatory subcortical input
		- **Fig. 2**** Strength of recurrent connections ****w**** and subcortical inputs ****I**** in 68 anatomically defined ROIs and their relationships with seven resting-state networks.**
			- ![](https://www.science.org/cms/10.1126/sciadv.aat7854/asset/27b800ca-4a8f-4913-8827-cdabaa74f5c0/assets/graphic/aat7854-f2.jpeg)
		- 脑区划分与模块划分的边界并不完全一致，因此需要构建一定的映射规则。
	- ### Regions with high recurrent connection strength are involved in sensory-motor processing, while those with low recurrent connection strength are involved in higher cognitive functions
		- 选择元分析中的12种认知成分。将68个ROI划分为周期性连接从低到高的10个区域，计算12种认知成分在各个区域的平均的归一化激活强度，结果表明周期性连接搞得区域偏向于在感觉、运动任务中激活，而周期性连接低的区域偏向于在认知功能相关的任务中激活。
		- **Fig. 3**** Relationship between recurrent connection strength ****w**** and BrainMap cognitive components.**
			- ![](https://www.science.org/cms/10.1126/sciadv.aat7854/asset/a1336157-e1e9-4a3b-bb3d-485719d744cb/assets/graphic/aat7854-f3.jpeg)
	- ### Strength of recurrent connections and subcortical inputs is associated with the first principal connectivity gradient
		- RSFC第一主梯度的两端是感觉运动区和默认活动网络，被认为是皮层处理层次的一个代表。在68个ROI中分别计算第一主梯度的平均值，并与皮质下输入和周期性连接强度做相关。主梯度与周期性连接强相关，与皮质下输入弱相关。
		- **Fig. 4**** Associations of estimated rMFM parameters (strength of recurrent connection ****w**** and subcortical input ****I****) with the first principal RSFC gradient and relative myelin content.**
			- ![](https://www.science.org/cms/10.1126/sciadv.aat7854/asset/f35b3a5a-3b0a-4a03-bda1-281126ec22a6/assets/graphic/aat7854-f4.jpeg)
	- ### Recurrent connection strength is positively associated with estimates of cortical myelin
		- T1w/T2w 比值已被广泛用于估计相对的皮质髓鞘含量（A multi-modal parcellation of human cerebral cortex | Nature）。（Hierarchy of transcriptomic specialization across human cortex captured by structural neuroimaging topography | Nature Neuroscience）认为T1w/T2w髓鞘估计值是很好的微观皮层处理层次的宏观代表。我们同样计算每个ROI的髓鞘估计平均值，并于周期性连接和皮层下输入做相关，周期性连接与髓鞘含量正相关，皮层下输入则无相关。
	- ### Strength of recurrent connections is positively associated with increased neuronal density
		- van den Heuvel 等人（Bridging Cytoarchitectonics and Connectomics in Human Cerebral Cortex | Journal of Neuroscience） 基于（ von Economo C F, Koskinas G N. Die cytoarchitektonik der hirnrinde des erwachsenen menschen[M]. J. Springer, 1925.  ）的细胞结构|cyto-architectonic工作，完成了对68个ROI的神经元大小和密度的估计。同样的，我们进行了相关性分析。需要指出，由于细胞结构的工作并未区分左右脑，故而我们先对参数在左右脑的对称区做平均，而后再进行相关性分析。
		- 周期性连接强度 w 与所有皮层的神经元密度平均值相关(r = 0.55，p = 0.00071) ，周期性连接强度 w 与皮层2、3、6层神经元密度呈显著相关。而与神经元大小无关。兴奋性皮层下输入与神经元密度或大小无关。
		- **Fig. 5**** Associations between estimated rMFM parameters (strength of recurrent connection ****w**** and subcortical input ****I****) and cytoarchitectonic measures (neuronal density and neuronal size) averaged across all cortical layers.**
			- ![](https://www.science.org/cms/10.1126/sciadv.aat7854/asset/1d93bbad-45a0-41b1-b1d0-c875ece2592f/assets/graphic/aat7854-f5.jpeg)
		- **Table 1**** Pearson’s correlation between estimated rMFM parameters (recurrent connection ****w**** and subcortical input ****I****) and cytoarchitectonic data (neuronal cell density and cell size).**
			- |   | ***w*** | ***P*** | ***I*** | ***P*** |
			  | ---- | ---- | ---- |
			  | Layer 1 density | −0.13 | 0.48 | 0.34 | 0.053 |
			  | Layer 2 density | **0.50** | **0.0038** | 0.23 | 0.21 |
			  | Layer 3 density | **0.52** | **0.0015** | 0.078 | 0.66 |
			  | Layer 4 density | 0.39 | 0.036 | −0.10 | 0.60 |
			  | Layer 5 density | −0.11 | 0.54 | 0.24 | 0.17 |
			  | Layer 6 density | **0.56** | **0.00054** | 0.23 | 0.20 |
			  | Cell density averaged across all layers | **0.55** | **0.00071** | 0.050 | 0.78 |
			  | Layer 1 size | −0.29 | 0.090 | 0.026 | 0.88 |
			  | Layer 2 size | −0.26 | 0.15 | −0.17 | 0.37 |
			  | Layer 3 size | 0.20 | 0.25 | −0.042 | 0.81 |
			  | Layer 4 size | 0.31 | 0.10 | −0.052 | 0.79 |
			  | Layer 5 size | 0.26 | 0.14 | 0.13 | 0.46 |
			  | Layer 6 size | −0.18 | 0.30 | −0.18 | 0.31 |
			  | Cell size averaged across all layers | 0.19 | 0.28 | 0.031 | 0.86 |
	- ### Replication with a higher-resolution parcellation and other control analyses
		- 使用不同的脑区分割方式进行计算。
		-
-
- ## Discussion
	- 关于灵长类大脑皮层中的层次结构已有丰富的论述（Saliency, switching, attention and control: a network model of insula function | SpringerLink；Multi-task connectivity reveals flexible hubs for adaptive task control | Nature Neuroscience ；）传统观点认为感觉信息流终止于额叶，但也有观点认为distributed association networks包括额叶、顶叶、颞叶、扣带同为最高层次。
	- 最近的文献主要支持联合网络的观点，但也指出联合网络内部可能存在层次差异。（Saliency, switching, attention and control: a network model of insula function | SpringerLink）认为额顶网络控制了不同任务的切换；（Multi-task connectivity reveals flexible hubs for adaptive task control | Nature Neuroscience）认为突显网络参与了状态的切换；（Situating the default-mode network along a principal gradient of macroscale cortical organization | PNAS）认为默认活动网络是最高级的区域，可以处理直接感觉无关的抽象信息。
	- 我们的结果认为默认活动网络和感觉运动系统位于层次的两端，并且提供了微观上的进一步分析。
	- 周期性连接强度将视觉、躯体运动、背侧注意和突显/腹侧注意网络  和  边缘系统、控制系统和默认网络区分开来，而皮层下输入则将后三者区分开来，两个指标是互补的。
	- ### Large-scale gradients of recurrent connection strength and subcortical inputs
		- 感觉运动区的强周期性连接及其与神经元大小的相关关系可能与其功能分化specialized local processing有关（Why is Brain Size so Important:Design Problems and Solutions as Neocortex Gets Biggeror Smaller | SpringerLink；Neuron densities vary across and within cortical areas in primates | PNAS）。其强皮层下输入可能与来自外界感知的信息流有关。
		- 默认活动网络的弱皮层下输入可能说明其缺乏直接来自外部的信息流，这与默认活动网络参与自我产生的思想，如自传性记忆、走神和思考未来等功能的假说相吻合（https://nyaspubs.onlinelibrary.wiley.com/doi/full/10.1111/nyas.12360?saml_referrer）。而若强循环连接对于特异化的局部信息处理非常重要，那么弱循环连接可能与默认网络假定的作为跨模态信息集成中心的角色一致。
		- 注意网络、边缘网络、控制网络位于层次结构的中间，但综合考虑周期性连接、皮层下输入两个参数，可以将它们进一步区分。注意网络相比另外二者有更高的周期性连接，控制网络相比另外二者有更弱的皮层下输入。我们可以将注意网络解释为中间层次的感觉运动处理系统（https://www.nature.com/articles/nrn755）
	- ### Model realism
		- 统计学方法(如 k 均值、 ICA、时间 ICA、滑动窗口相关性和隐马尔可夫模型)已被广泛用于研究脑网络的组织和动力学(The dynamic functional connectome: State-of-the-art and perspectives - ScienceDirect)。虽然这些模型为人类大脑提供了重要的洞察力，但它们并不是为了“模仿”实际的大脑机制。  生物物理模型一方面参考了微观尺度的环路机制，纳入了微观参数，另一方面能够对宏观数据进行一定程度的模拟，可能是沟通微观机制与宏观测量的有效方法（Dynamic models of large-scale brain activity | Nature Neuroscience）。
		- MFM模型是对脉冲神经网络的简化，但要比DCM更具备生物意义。平均场模型类似于统计物理的方法，对神经元群体进行统计平均，这虽然会损失信息，但更容易与fMRI的宏观测量相结合。更真实更细致的神经模型未必能更好地回答神经科学问题。
		- 将兴奋性连接和抑制性连接区分开来，可能是MFM发展的下一个目标（How Local Excitation–Inhibition Ratio Impacts the Whole Brain Dynamics | Journal of Neuroscience；Just a moment...）。
	- ### Model fit
		- 目前，没有统一的标准对不同文献提出的模型的拟合优度进行比较。
-
- ## Materials and Methods
	- ### Data
		- HCP S500 release  中的452个被试。
		- %LOCAL_FILE%o1VJoePiaoCvV4ND-U7JB94bHY-zDVVFkMSHayyfE1TzlFemDM-lsxq4T89WyN9c6yo8h0TKtdL7bBOZ2G-brSrgippu_JzGXvqHGAIqtD-mUvJd_xNymbPC9KZKeYYk.png
	- ### Dynamic MFM
		- %LOCAL_FILE%ziMw9XUBqBxsaV0C1XGIHyjTwJ-8qCDhSQeWrvbD4xNSXDVTNaQQWsKD91NFnO12TutqOSmn6zw0zneSGAGToBmHSPbj76aXN0NVkDtLSAtCGWMnQOKHIrM5YJQ-g8vG.png  %LOCAL_FILE%xq5_ZZ0BeneVYbWKtHm5EZnQsxtT6pbIfPlaTrYCwsJndZBCKr5ttw2DjmGbNHBfT7rpQACrIZ1SGutklG6mx9rPUk_zRgU80ooQT0dVVWxe8Y_zB6RQTYz9foqGSSYe.png
		- H(x) 是一个整流线性单元的连续近似。
		- 上述为脑区i的状态演化方程，是一个非线性的随机微分方程。S_i 表示average synaptic gating variable，x_i 表示总输入电流，H(x_i) 表示群体发射率population firing rate。可以看到，周期性连接、脑区间连接和皮层下输入决定了总输入电流。
		- G、w作为比例系数，调节三个电流组分的相对大小，J=0.2609nA为synaptic coupling，a=270n/C，b=108Hz，d=0.154s，\tau_s=0.1s，v_i(t) 为独立标准高斯噪声。
		- average synaptic gating variable是度量突触后离子通道开关比例的变量，开关比例越大，该区域内发生的电荷流动就越强。此处使用的是带噪声的一阶动力学First-order kinetics（参见Synaptic dynamics)
		- MFM 采用可解释的动态变量和生理参数，如总体放电率和平均突触门控变量，在神经元群体水平上捕捉皮层区域的平均神经动态行为。与 AMPA、 NMDA (n- 甲基 -d- 天冬氨酸)和 GABA (γ- 氨基丁酸)突触耦合的更详细的神经元网络相比，MFM 可以以相对较低的参数复杂度全面研究模型参数和大规模脑动力学之间的关系。
		- 神经活动S_i 通过Balloon-Windkessel的血流动力学（Dynamic causal modelling - ScienceDirect）  转换为BOLD信号。
			- %LOCAL_FILE%bnP4FiJNYu8bP1HaXV3SM2Sd3hM656QFRK_mYx7ghnl24L2kLMyKRCGJeF2shgEQv1a51iXUxMYQIE8runfHg1rqIQZq3JVJzSQrQof9AAj6eRe5uZY5ZV3uYKWBNnUl.png
			- %LOCAL_FILE%ftHmrrQkAOBD5wLP_llqWr4G_GVhRK1tAW0BCH1JO7KPlMW02inPcUS1lmBB_JAZLR7FzJ9LR9wO2Qxb314wg8-1PO-8sm7kN82Wc_h8AqwgBnXjbcp4pzOoiZAgzxmK.png
			- %LOCAL_FILE%kIDgZ-mkl_BSP_ha3JZ3Hc9oHS_Inum5CJE8MG24PeaKbkX9f87WK_qjFaIX2jnTQgRHjzb2mufVFsd6CRH9_mYayze4TR2mTjGSXqf_WkGI0HM8ywjZFcsMDOpDpq5m.png
		- 简单地说，神经活动S引发一个血管舒张信号z，流入f与z成比例，并会带动血液容量v和脱氧血红蛋白含量q的变化。其中参数\rho=0.34,\kappa=0.65s^{-1},\tau=0.98s,\alpha=0.32,\gamma=0.41s^{-1}
		- BOLD信号由血容量和脱氧血红蛋白含量得出，V_0=0.02 是对应于静息态的resting blood volume fraction， 参数组k是对应于磁共振的测量参数。
		- 使用欧拉积分法进行模拟，时间步长为10ms，S值随机初始化，模拟16.4分钟并舍弃前2分钟，BOLD信号被下采样至0.72s。
	- ### rMFM and automatic estimation of model parameters
		- 使用DCM中（Dynamic causal modelling - ScienceDirect）的期望最大化算法对参数进行优化。算法如下：
			- Step 1：
				- 初始化模型参数\theta_0=[w_i,I_i,G,\sigma]^T=[0.5,0.3,1,0.001]
			- Step 2:
				- https://remnote-user-data.s3.amazonaws.com/KoXNQSwfQy6UKcfvFfTnpA_CvZicqJCXtm9mondJcN1etfrjrzQfEV3I0Cmd-gg0pOiGgsyUaDslQBzjIFcur8rr_m2LC8vF_EpLxV5zExzNUq9txQ_AfHCN98syem9j.png
			- Step 3:
				- https://remnote-user-data.s3.amazonaws.com/li0wtAIMwF6uSauqQnQMNFQm3lMsLX0gkPT9UVUZhjVPOHr9eXX-WRDH9OGfqzeajTXSP0t71kOACF6V4X9c161T7buqACBD7ffa8Ud1uyKT9c0K3xUT-WD9eExabhC2.png
			- Step 4:
				- https://remnote-user-data.s3.amazonaws.com/Ae8X0DRvMaDSOw__4ZeqlD6TaniQd7j7M1piJEuIwxRsqGk3EbkmaXFBY35teQpKxP1qfcp8EtwQ6ag-EpIL_FHlYHUrrhC4dGRFZomf2qJP4_qSCDlLlHmT5cX-ws9b.png
			- Step 5:
				- https://remnote-user-data.s3.amazonaws.com/ykk2qbrgYT_sltk5luvKECHbf4OBcMNiKR6xFUOe2eMNybBfha3SxGsxPNVMqUiwRM21byQKjbrrxNzrp8dOhB_9WYpzTf2LmosFanOjYnzKAobcrcPvccll60p90s5b.png
			- Step 6:
				- https://remnote-user-data.s3.amazonaws.com/ORZscMQxhHr39HcVgy1QDWhOXXQyJsnBA4qqPJDc3Zxoh9hWarVGLFH6DRbGgTSt3Epen8eLhLNsrO7Z3v7L0SstqjuvtbxlyvGpaBnk5RvGPzsaL3fFdTrKyUcKDIz4.png
			- Step 7:
				- https://remnote-user-data.s3.amazonaws.com/AML6fR1QQaKhndlAjTuk1j7aQUdXKtm8z7WDvMtmCFPUfsoHcZq8zgLHpp8YN6FKpoBhdecq4rR1Xi8gS6THm-dDQ4QoU-sr3iEAi4gXwGubm2x1oHZBkPVFSjmyGMif.png
				- λ的初值为-3
			- Step 8：
				- https://remnote-user-data.s3.amazonaws.com/c96lGRlnNBdY5we50yB3lVpFttl1NGNHvy7pEIRVATNL488vbbdDc_LQFsDHiJnsbFzQyy0vtjiDOGUJ8PVzuKdycVWU5LRZsqQuj-9qJ3rtsjSUmgsKaHl0EKAoheXf.png
			- Step 9：循环
	- 几点注意：
		- 使用68脑区分割是延续之前的MFM研究的做法，功能磁共振所用的脑区分割在形状上并不规则，对于纤维重构步骤并不稳定。
		- 上述优化算法既看不懂也过于耗时……考虑使用遗传算法作为替代。
		- 使用贝叶斯（甚至分层贝叶斯）方法估计个体网络是否可行？