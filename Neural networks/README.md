
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Лабораторные-работы-5-8">Лабораторные работы 5-8<a class="anchor-link" href="#%D0%9B%D0%B0%D0%B1%D0%BE%D1%80%D0%B0%D1%82%D0%BE%D1%80%D0%BD%D1%8B%D0%B5-%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%8B-5-8">¶</a></h1><h2 id="Нейронные-сети">Нейронные сети<a class="anchor-link" href="#%D0%9D%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D1%8B%D0%B5-%D1%81%D0%B5%D1%82%D0%B8">¶</a></h2><p>Для начала импортируем модуль <strong>numpy</strong> для удобного хранения и обработки данных</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>

<span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">seed</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Далее опишим некоторые из функций активации (такие как <strong>relu</strong>, <strong>sigmoid</strong>, <strong>tanh</strong> и тп), которые будут использоваться в дальнейшем.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">relu</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">rev</span><span class="o">=</span><span class="kc">False</span><span class="p">):</span>
    <span class="k">if</span> <span class="ow">not</span> <span class="n">rev</span><span class="p">:</span>
        <span class="k">return</span> <span class="p">(</span><span class="n">x</span> <span class="o">&gt;</span> <span class="mi">0</span><span class="p">)</span> <span class="o">*</span> <span class="n">x</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="k">return</span> <span class="n">x</span> <span class="o">&gt;</span> <span class="mi">0</span>

    
<span class="k">def</span> <span class="nf">sigmoid</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">rev</span><span class="o">=</span><span class="kc">False</span><span class="p">):</span>
    <span class="k">if</span> <span class="ow">not</span> <span class="n">rev</span><span class="p">:</span>
        <span class="k">return</span> <span class="mi">1</span> <span class="o">/</span> <span class="p">(</span><span class="mi">1</span> <span class="o">+</span> <span class="n">np</span><span class="o">.</span><span class="n">exp</span><span class="p">(</span><span class="o">-</span><span class="n">x</span><span class="p">))</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="k">return</span> <span class="n">sigmoid</span><span class="p">(</span><span class="n">x</span><span class="p">)</span> <span class="o">*</span> <span class="p">(</span><span class="mi">1</span> <span class="o">-</span> <span class="n">sigmoid</span><span class="p">(</span><span class="n">x</span><span class="p">))</span>

    
<span class="k">def</span> <span class="nf">tanh</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">rev</span><span class="o">=</span><span class="kc">False</span><span class="p">):</span>
    <span class="k">if</span> <span class="ow">not</span> <span class="n">rev</span><span class="p">:</span>
        <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">tanh</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="k">return</span> <span class="mi">1</span> <span class="o">-</span> <span class="p">(</span><span class="n">x</span> <span class="o">**</span> <span class="mi">2</span><span class="p">)</span>


<span class="k">def</span> <span class="nf">liner</span><span class="p">(</span><span class="n">x</span><span class="p">):</span>
    <span class="k">return</span> <span class="n">x</span>


<span class="k">def</span> <span class="nf">softmax</span><span class="p">(</span><span class="n">x</span><span class="p">):</span>
    <span class="n">temp</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">exp</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">temp</span> <span class="o">/</span> <span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">temp</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">keepdims</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Теперь напишим основной класс для нейронной сети, который бует использоваться для решения последующих работ.<br>
Данный класс будет иметь параметры:</p>
<ul>
<li>dataset - сами данные на основе которых будет производится обучения</li>
<li>answer - правильные ответы для входных данных, так же используются для обучения</li>
<li>neural_count - колличество нейронов в скрытом слое</li>
<li>iteration_count - колличесвто эпох обучения</li>
<li>alpha - коэффициент используемый для уменьшения шага весов при обучении</li>
<li>hidden_layer_activation_func - функция активации для скрытого слоя</li>
<li>output_layer_activation_func - функция активации для выходного слоя</li>
<li>weights_0_1 - веса скрытого слоя</li>
<li>weights_1_2 - веса выходного слоя</li>
</ul>
<p>А так же методы:</p>
<ul>
<li>generate_weights - генерирует случайные веса для данной сети</li>
<li>train - основной метод обучения нерйронной сети</li>
<li>predict - делает предсказание используя веса уже обученной сети</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">class</span> <span class="nc">NeuralNetwork</span><span class="p">:</span>
    <span class="k">def</span> <span class="nf">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">dataset</span><span class="p">,</span> <span class="n">answer</span><span class="p">,</span> <span class="n">neural_count</span><span class="o">=</span><span class="mi">6</span><span class="p">,</span> <span class="n">iteration_count</span><span class="o">=</span><span class="mi">10000</span><span class="p">,</span> <span class="n">alpha</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span> <span class="n">func1</span><span class="o">=</span><span class="n">relu</span><span class="p">,</span> <span class="n">func2</span><span class="o">=</span><span class="n">liner</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">dataset</span> <span class="o">=</span> <span class="n">dataset</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">answer</span> <span class="o">=</span> <span class="n">answer</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">neural_count</span> <span class="o">=</span> <span class="n">neural_count</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">iteration_count</span> <span class="o">=</span> <span class="n">iteration_count</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">alpha</span> <span class="o">=</span> <span class="n">alpha</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">hidden_layer_activation_func</span> <span class="o">=</span> <span class="n">func1</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">output_layer_activation_func</span> <span class="o">=</span> <span class="n">func2</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">weights_0_1</span> <span class="o">=</span> <span class="mi">0</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">weights_1_2</span> <span class="o">=</span> <span class="mi">0</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">GenerateWeights</span><span class="p">()</span>

    <span class="k">def</span> <span class="nf">GenerateWeights</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">weights_0_1</span> <span class="o">=</span> <span class="mi">2</span><span class="o">*</span><span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">sample</span><span class="p">((</span><span class="nb">len</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">dataset</span><span class="p">[</span><span class="mi">0</span><span class="p">]),</span> <span class="bp">self</span><span class="o">.</span><span class="n">neural_count</span><span class="p">))</span> <span class="o">-</span> <span class="mi">1</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">weights_1_2</span> <span class="o">=</span> <span class="mi">2</span><span class="o">*</span><span class="n">np</span><span class="o">.</span><span class="n">random</span><span class="o">.</span><span class="n">sample</span><span class="p">((</span><span class="bp">self</span><span class="o">.</span><span class="n">neural_count</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">answer</span><span class="p">[</span><span class="mi">0</span><span class="p">])))</span> <span class="o">-</span> <span class="mi">1</span>
        
    <span class="k">def</span> <span class="nf">Train</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">dropout</span><span class="o">=</span><span class="kc">False</span><span class="p">):</span>
        <span class="kn">from</span> <span class="nn">time</span> <span class="k">import</span> <span class="n">time</span>
        <span class="n">start</span> <span class="o">=</span> <span class="n">time</span><span class="p">()</span>
        <span class="k">for</span> <span class="n">iteration</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">iteration_count</span><span class="p">):</span>
            <span class="n">layer_2_error</span> <span class="o">=</span> <span class="mi">0</span>
            <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">dataset</span><span class="p">)):</span>
                <span class="bp">self</span><span class="o">.</span><span class="n">layer_0</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">dataset</span><span class="p">[</span><span class="n">i</span><span class="p">:</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">]</span>
                <span class="bp">self</span><span class="o">.</span><span class="n">Predict</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">layer_0</span><span class="p">)</span>
                <span class="n">layer_2_error</span> <span class="o">+=</span> <span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">((</span><span class="bp">self</span><span class="o">.</span><span class="n">layer_2</span> <span class="o">-</span> <span class="bp">self</span><span class="o">.</span><span class="n">answer</span><span class="p">[</span><span class="n">i</span><span class="p">:</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">])</span> <span class="o">**</span> <span class="mi">2</span><span class="p">)</span>
                
                <span class="k">if</span> <span class="n">sigmoid</span> <span class="ow">in</span> <span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">hidden_layer_activation_func</span><span class="p">,</span> <span class="bp">self</span><span class="o">.</span><span class="n">output_layer_activation_func</span><span class="p">):</span>
                    <span class="bp">self</span><span class="o">.</span><span class="n">ce</span><span class="p">(</span><span class="n">i</span><span class="p">)</span>
                <span class="k">else</span><span class="p">:</span>
                    <span class="bp">self</span><span class="o">.</span><span class="n">mse</span><span class="p">(</span><span class="n">i</span><span class="p">)</span>

            <span class="k">if</span> <span class="n">iteration</span> <span class="o">%</span> <span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">iteration_count</span> <span class="o">//</span> <span class="mi">10</span><span class="p">)</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
                <span class="nb">print</span><span class="p">(</span><span class="s1">'На итерации'</span><span class="p">,</span> <span class="n">iteration</span><span class="p">,</span> <span class="s1">'ошибка -'</span><span class="p">,</span> <span class="n">layer_2_error</span><span class="p">)</span>

        <span class="nb">print</span><span class="p">(</span><span class="s1">'_'</span> <span class="o">*</span> <span class="mi">50</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="s1">'Итоговая ошибка'</span><span class="p">,</span> <span class="n">layer_2_error</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">()</span>
        <span class="nb">print</span><span class="p">(</span><span class="s1">'Обучение заняло'</span><span class="p">,</span> <span class="n">time</span><span class="p">()</span><span class="o">-</span><span class="n">start</span><span class="p">,</span> <span class="s1">'сек'</span><span class="p">)</span>

    <span class="k">def</span> <span class="nf">mse</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">i</span><span class="p">):</span>
        <span class="n">layer_2_delta</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">layer_2</span> <span class="o">-</span> <span class="bp">self</span><span class="o">.</span><span class="n">answer</span><span class="p">[</span><span class="n">i</span><span class="p">:</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">]</span>
        <span class="n">layer_1_delta</span> <span class="o">=</span> <span class="n">layer_2_delta</span><span class="o">.</span><span class="n">dot</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">weights_1_2</span><span class="o">.</span><span class="n">T</span><span class="p">)</span> \
                                <span class="o">*</span> <span class="bp">self</span><span class="o">.</span><span class="n">hidden_layer_activation_func</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">layer_1</span><span class="p">,</span> <span class="n">rev</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
        
        <span class="bp">self</span><span class="o">.</span><span class="n">weights_1_2</span> <span class="o">-=</span> <span class="bp">self</span><span class="o">.</span><span class="n">alpha</span> <span class="o">*</span> <span class="bp">self</span><span class="o">.</span><span class="n">layer_1</span><span class="o">.</span><span class="n">T</span><span class="o">.</span><span class="n">dot</span><span class="p">(</span><span class="n">layer_2_delta</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">weights_0_1</span> <span class="o">-=</span> <span class="bp">self</span><span class="o">.</span><span class="n">alpha</span> <span class="o">*</span> <span class="bp">self</span><span class="o">.</span><span class="n">layer_0</span><span class="o">.</span><span class="n">T</span><span class="o">.</span><span class="n">dot</span><span class="p">(</span><span class="n">layer_1_delta</span><span class="p">)</span>
        
    <span class="k">def</span> <span class="nf">ce</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">i</span><span class="p">):</span>
        <span class="n">layer_2_delta</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">dot</span><span class="p">(</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">layer_1</span><span class="o">.</span><span class="n">T</span><span class="p">,</span>
            <span class="p">(</span><span class="mi">2</span> <span class="o">*</span> <span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">answer</span><span class="p">[</span><span class="n">i</span><span class="p">:</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">]</span> <span class="o">-</span> <span class="bp">self</span><span class="o">.</span><span class="n">layer_2</span><span class="p">)</span> <span class="o">*</span> <span class="bp">self</span><span class="o">.</span><span class="n">output_layer_activation_func</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">layer_2</span><span class="p">,</span> <span class="n">rev</span><span class="o">=</span><span class="kc">True</span><span class="p">))</span>
        <span class="p">)</span>

        <span class="n">layer_1_delta</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">dot</span><span class="p">(</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">layer_0</span><span class="o">.</span><span class="n">T</span><span class="p">,</span>
            <span class="n">np</span><span class="o">.</span><span class="n">dot</span><span class="p">(</span>
                <span class="mi">2</span><span class="o">*</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">answer</span><span class="p">[</span><span class="n">i</span><span class="p">:</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">]</span> <span class="o">-</span> <span class="bp">self</span><span class="o">.</span><span class="n">layer_2</span><span class="p">)</span> <span class="o">*</span> <span class="bp">self</span><span class="o">.</span><span class="n">output_layer_activation_func</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">layer_2</span><span class="p">,</span> <span class="n">rev</span><span class="o">=</span><span class="kc">True</span><span class="p">),</span>
                <span class="bp">self</span><span class="o">.</span><span class="n">weights_1_2</span><span class="o">.</span><span class="n">T</span><span class="p">)</span> <span class="o">*</span> <span class="bp">self</span><span class="o">.</span><span class="n">hidden_layer_activation_func</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">layer_1</span><span class="p">,</span> <span class="n">rev</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
        <span class="p">)</span>

        <span class="bp">self</span><span class="o">.</span><span class="n">weights_1_2</span> <span class="o">+=</span> <span class="bp">self</span><span class="o">.</span><span class="n">alpha</span> <span class="o">*</span> <span class="n">layer_2_delta</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">weights_0_1</span> <span class="o">+=</span> <span class="bp">self</span><span class="o">.</span><span class="n">alpha</span> <span class="o">*</span> <span class="n">layer_1_delta</span>

    <span class="k">def</span> <span class="nf">Predict</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">data</span><span class="p">,</span> <span class="n">dropout</span><span class="o">=</span><span class="kc">False</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">layer_1</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">hidden_layer_activation_func</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">dot</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="bp">self</span><span class="o">.</span><span class="n">weights_0_1</span><span class="p">))</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">layer_2</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">output_layer_activation_func</span><span class="p">(</span><span class="n">np</span><span class="o">.</span><span class="n">dot</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">layer_1</span><span class="p">,</span> <span class="bp">self</span><span class="o">.</span><span class="n">weights_1_2</span><span class="p">))</span>

        <span class="k">if</span> <span class="ow">not</span> <span class="p">(</span><span class="n">data</span> <span class="ow">is</span> <span class="bp">self</span><span class="o">.</span><span class="n">layer_0</span><span class="p">):</span>
            <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">layer_2</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Лабораторная-работа-5">Лабораторная работа 5<a class="anchor-link" href="#%D0%9B%D0%B0%D0%B1%D0%BE%D1%80%D0%B0%D1%82%D0%BE%D1%80%D0%BD%D0%B0%D1%8F-%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0-5">¶</a></h1><h2 id="Создание-и-обучение-простейшей-нейронной-сети">Создание и обучение простейшей нейронной сети<a class="anchor-link" href="#%D0%A1%D0%BE%D0%B7%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5-%D0%B8-%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5-%D0%BF%D1%80%D0%BE%D1%81%D1%82%D0%B5%D0%B9%D1%88%D0%B5%D0%B9-%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D0%BE%D0%B9-%D1%81%D0%B5%D1%82%D0%B8">¶</a></h2><p><strong>Цель</strong> - создание и обучение простейшей нейронной сети для решения задачи XOR для двух элементов.<br>
Для начала в переменные <strong><em>xor</em></strong> и <strong><em>xor_answer</em></strong> записываем данные на которых будет производится обучение модели.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">xor</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([[</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">],</span>
                <span class="p">[</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span>
                <span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">],</span>
                <span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">]])</span>
<span class="n">xor_answer</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([[</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">]])</span><span class="o">.</span><span class="n">T</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Далее создаем объект Нейронной сети для данной задачи. Будем использовать 4 нейрона в скрытом слое и 1000 эпох обучения.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Xor</span> <span class="o">=</span> <span class="n">NeuralNetwork</span><span class="p">(</span><span class="n">dataset</span><span class="o">=</span><span class="n">xor</span><span class="p">,</span>
                    <span class="n">answer</span><span class="o">=</span><span class="n">xor_answer</span><span class="p">,</span>
                    <span class="n">alpha</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
                    <span class="n">neural_count</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
                    <span class="n">iteration_count</span><span class="o">=</span><span class="mi">500</span><span class="p">)</span>
<span class="n">Xor</span><span class="o">.</span><span class="n">GenerateWeights</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>После случайной генерации веса имеют значения:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Xor</span><span class="o">.</span><span class="n">weights_0_1</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[6]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>array([[-0.5910955 ,  0.75623487, -0.94522481,  0.34093502],
       [-0.1653904 ,  0.11737966, -0.71922612, -0.60379702]])</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Xor</span><span class="o">.</span><span class="n">weights_1_2</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[7]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>array([[ 0.60148914],
       [ 0.93652315],
       [-0.37315164],
       [ 0.38464523]])</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Произведем обучение модели.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Xor</span><span class="o">.</span><span class="n">Train</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>На итерации 0 ошибка - 1.6825749810337902
На итерации 50 ошибка - 0.27516577442941104
На итерации 100 ошибка - 2.1357415879146058e-05
На итерации 150 ошибка - 1.8484091669274243e-10
На итерации 200 ошибка - 1.5791204071587323e-15
На итерации 250 ошибка - 1.3490112339976078e-20
На итерации 300 ошибка - 1.1525550683090072e-25
На итерации 350 ошибка - 1.8504634031807123e-30
На итерации 400 ошибка - 4.6995681904394146e-31
На итерации 450 ошибка - 4.6995681904394146e-31
__________________________________________________
Итоговая ошибка 4.6995681904394146e-31

Обучение заняло 0.07262253761291504 сек
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Теперь веса в нашей модели имеют значения:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Xor</span><span class="o">.</span><span class="n">weights_0_1</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[9]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>array([[-0.5910955 , -0.82800213, -0.94522481,  0.89473936],
       [-0.1653904 ,  0.82800213, -0.71922612, -0.90029912]])</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Xor</span><span class="o">.</span><span class="n">weights_1_2</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[10]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>array([[ 0.60148914],
       [ 1.20772637],
       [-0.37315164],
       [ 1.11764391]])</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Протестируем работу обученной модели, посмотрим какой результат она выдаст на наши входные данные.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s1">'По итогу обучения модель считает:'</span><span class="p">)</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">xor</span><span class="p">)):</span>
    <span class="n">data</span> <span class="o">=</span> <span class="n">xor</span><span class="p">[</span><span class="n">i</span><span class="p">:</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">]</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">f</span><span class="s1">'</span><span class="si">{data[0][0]}</span><span class="s1"> XOR </span><span class="si">{data[0][1]}</span><span class="s1"> = {round(Xor.Predict(data)[0][0])}'</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>По итогу обучения модель считает:
0 XOR 0 = 0
0 XOR 1 = 1
1 XOR 0 = 1
1 XOR 1 = 0
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong><em>Вывод</em></strong><br>
В данной работе нам удалось построить свою первую нейронную сеть состояющую из 4 нейроннов скрытого слоя и одного выходного, которая решает задачу <em>XOR</em> между 0 и 1. По итогам теста данная сеть на 100 справилась со своей задачей, а итоговая ошибка составила всего лишь 4.6995681904394146e-31.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Лабораторная-работа-6">Лабораторная работа 6<a class="anchor-link" href="#%D0%9B%D0%B0%D0%B1%D0%BE%D1%80%D0%B0%D1%82%D0%BE%D1%80%D0%BD%D0%B0%D1%8F-%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0-6">¶</a></h1><h2 id="Определение-напрвления-двоичного-сдвига">Определение напрвления двоичного сдвига<a class="anchor-link" href="#%D0%9E%D0%BF%D1%80%D0%B5%D0%B4%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5-%D0%BD%D0%B0%D0%BF%D1%80%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D1%8F-%D0%B4%D0%B2%D0%BE%D0%B8%D1%87%D0%BD%D0%BE%D0%B3%D0%BE-%D1%81%D0%B4%D0%B2%D0%B8%D0%B3%D0%B0">¶</a></h2><p><strong>Цель</strong> - построение, обучение и тестирование нейронной сети, предназначенной для определения направления двоичного кода.<br>
Первое что необходимо сделать, это сгенерировать входные данные, которые будут представлять из себя матрицу. В данной матрице строка, это изначльные нули и еденицы, в колличестве равном значению <em>count</em> (5 штук), а так же их представление уже со сдвигом.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">count</span> <span class="o">=</span> <span class="mi">5</span>

<span class="n">binary</span> <span class="o">=</span> <span class="p">[</span><span class="nb">list</span><span class="p">(</span><span class="nb">bin</span><span class="p">(</span><span class="n">i</span><span class="p">)[</span><span class="mi">2</span><span class="p">:]</span><span class="o">.</span><span class="n">zfill</span><span class="p">(</span><span class="mi">5</span><span class="p">))</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="o">**</span><span class="n">count</span><span class="o">-</span><span class="mi">1</span><span class="p">)]</span>
<span class="n">binary</span> <span class="o">=</span> <span class="p">[</span><span class="n">i</span> <span class="o">+</span> <span class="n">i</span><span class="p">[</span><span class="mi">1</span><span class="p">:]</span> <span class="o">+</span> <span class="n">i</span><span class="p">[</span><span class="mi">0</span><span class="p">:</span><span class="mi">1</span><span class="p">]</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">binary</span><span class="p">]</span> <span class="o">+</span> <span class="p">[</span><span class="n">i</span> <span class="o">+</span> <span class="n">i</span><span class="p">[</span><span class="o">-</span><span class="mi">1</span><span class="p">:]</span> <span class="o">+</span> <span class="n">i</span><span class="p">[:</span><span class="nb">len</span><span class="p">(</span><span class="n">i</span><span class="p">)</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">binary</span><span class="p">]</span>
<span class="n">binary</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">binary</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">int</span><span class="p">)</span>
<span class="n">binary_answer</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([[</span><span class="o">*</span><span class="p">[</span><span class="mi">0</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">2</span><span class="o">**</span><span class="n">count</span><span class="o">-</span><span class="mi">2</span><span class="p">)],</span> <span class="o">*</span><span class="p">[</span><span class="mi">1</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">2</span><span class="o">**</span><span class="n">count</span><span class="o">-</span><span class="mi">2</span><span class="p">)]</span> <span class="p">]])</span><span class="o">.</span><span class="n">T</span> 

<span class="n">binary</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[12]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>array([[0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
       [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
       [0, 0, 0, 1, 1, 0, 0, 1, 1, 0],
       [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
       [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
       [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
       [0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
       [0, 1, 0, 1, 1, 1, 0, 1, 1, 0],
       [0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
       [0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
       [0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
       [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [1, 0, 0, 0, 1, 0, 0, 0, 1, 1],
       [1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
       [1, 0, 0, 1, 1, 0, 0, 1, 1, 1],
       [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
       [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
       [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
       [1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
       [1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
       [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
       [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
       [1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
       [1, 1, 1, 0, 0, 1, 1, 0, 0, 1],
       [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
       [1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
       [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
       [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
       [0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
       [0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
       [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
       [0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
       [0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
       [0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
       [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
       [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
       [0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
       [0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
       [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
       [1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
       [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
       [1, 0, 0, 1, 1, 1, 1, 0, 0, 1],
       [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
       [1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
       [1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
       [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
       [1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
       [1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
       [1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
       [1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
       [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
       [1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
       [1, 1, 1, 1, 0, 0, 1, 1, 1, 1]])</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Увеличим колличество нейроннов в скрытом слое по сравнению с первой работой до 10, так как здесь поиск закономерности является более сложной задачей.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Binary</span> <span class="o">=</span> <span class="n">NeuralNetwork</span><span class="p">(</span><span class="n">dataset</span><span class="o">=</span><span class="n">binary</span><span class="p">,</span>
                       <span class="n">answer</span><span class="o">=</span><span class="n">binary_answer</span><span class="p">,</span>
                       <span class="n">neural_count</span><span class="o">=</span><span class="mi">10</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Произведем обучение модели.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Binary</span><span class="o">.</span><span class="n">Train</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>На итерации 0 ошибка - 18.97740014099182
На итерации 1000 ошибка - 0.5684430473562058
На итерации 2000 ошибка - 0.3275776557933158
На итерации 3000 ошибка - 0.22605799259031428
На итерации 4000 ошибка - 0.33339584387096827
На итерации 5000 ошибка - 0.00013536439910043454
На итерации 6000 ошибка - 1.3588924830368137e-06
На итерации 7000 ошибка - 1.5254175183160808e-08
На итерации 8000 ошибка - 1.496170081031347e-08
На итерации 9000 ошибка - 1.3966431377696958e-09
__________________________________________________
Итоговая ошибка 1.6991406033880625e-09

Обучение заняло 17.83565378189087 сек
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Протестируем работу обученной модели. Посмотрим какой результат она выдаст на наши входные данные.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[15]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">random</span> <span class="k">import</span> <span class="n">randint</span>


<span class="n">test</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="n">binary</span><span class="p">[</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="mi">59</span><span class="p">)]</span><span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">10</span><span class="p">)])</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">test</span><span class="p">)):</span>
    <span class="n">data</span> <span class="o">=</span> <span class="n">test</span><span class="p">[</span><span class="n">i</span><span class="p">:</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">]</span>
    <span class="n">ans</span> <span class="o">=</span>  <span class="s1">'вправо'</span> <span class="k">if</span> <span class="nb">round</span><span class="p">(</span><span class="n">Binary</span><span class="o">.</span><span class="n">Predict</span><span class="p">(</span><span class="n">data</span><span class="p">)[</span><span class="mi">0</span><span class="p">][</span><span class="mi">0</span><span class="p">])</span> <span class="k">else</span> <span class="s1">'влево'</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">f</span><span class="s1">'</span><span class="si">{data[0][:count]}</span><span class="s1"> -&gt; </span><span class="si">{data[0][count:]}</span><span class="s1"> - сдвиг </span><span class="si">{ans}</span><span class="s1">'</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>[0 1 0 0 1] -&gt; [1 0 0 1 0] - сдвиг влево
[1 1 0 0 1] -&gt; [1 1 1 0 0] - сдвиг вправо
[0 1 1 0 1] -&gt; [1 1 0 1 0] - сдвиг влево
[1 1 0 0 1] -&gt; [1 0 0 1 1] - сдвиг влево
[0 0 1 1 0] -&gt; [0 0 0 1 1] - сдвиг вправо
[1 1 0 0 1] -&gt; [1 0 0 1 1] - сдвиг влево
[1 0 0 0 1] -&gt; [1 1 0 0 0] - сдвиг вправо
[1 0 1 1 1] -&gt; [1 1 0 1 1] - сдвиг вправо
[1 0 1 0 0] -&gt; [0 1 0 1 0] - сдвиг вправо
[0 0 1 0 1] -&gt; [0 1 0 1 0] - сдвиг влево
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>Вывод</strong><br>
Данная нейросеть ищет более сложные закономероности, поэтому нуждается в большем колличестве нейронов. По итогом произведенного обучения итоговая ошибка составила 1.6991406033880625e-09, что является хорошим результатом. Значит после произведенного обучения нейронная сети отлично справляется с поставленной задачей. Это хорошо заметно на результатх теста.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Лабораторная-работа-7.1.">Лабораторная работа 7.1.<a class="anchor-link" href="#%D0%9B%D0%B0%D0%B1%D0%BE%D1%80%D0%B0%D1%82%D0%BE%D1%80%D0%BD%D0%B0%D1%8F-%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0-7.1.">¶</a></h1><h2 id="Распознавание-символов">Распознавание символов<a class="anchor-link" href="#%D0%A0%D0%B0%D1%81%D0%BF%D0%BE%D0%B7%D0%BD%D0%B0%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5-%D1%81%D0%B8%D0%BC%D0%B2%D0%BE%D0%BB%D0%BE%D0%B2">¶</a></h2><p><strong>Цель</strong> - разработать и исслодовать нейронную сеть обратного распределения предназначенную для распознавания образов.<br>
В переменные <strong><em>lyters</em></strong> и <strong><em>lyters_answer</em></strong> записываем данные на которых будет производится обучение модели.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">lyters</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([[</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span>
                   <span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">],</span>
                   <span class="p">[</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">],</span>
                   <span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">]])</span>

<span class="n">lyters_answer</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span>  <span class="c1"># X</span>
                          <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span>  <span class="c1"># Y</span>
                          <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span>  <span class="c1"># I</span>
                          <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">]])</span> <span class="c1"># C</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Создадим объект нейронной сети. В скрытом слою будем использовать 15 нейронов, а так же возьмем колличество эпох равное 20000. До этого случая у нас был один выход <em>(1 или 0)</em>, но сейчас нам нужно классифицировать 4 разные буквы, и выходов для этого мы будем использовать 4 соответвенно, поэтому здесь в качестве функции активации выходного слоя лучше использовать функцию softmax.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[17]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Lyters</span> <span class="o">=</span> <span class="n">NeuralNetwork</span><span class="p">(</span><span class="n">dataset</span><span class="o">=</span><span class="n">lyters</span><span class="p">,</span>
                       <span class="n">answer</span><span class="o">=</span><span class="n">lyters_answer</span><span class="p">,</span>
                       <span class="n">neural_count</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
                       <span class="n">func2</span><span class="o">=</span><span class="n">softmax</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Произведем обучение модели.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[18]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Lyters</span><span class="o">.</span><span class="n">Train</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>На итерации 0 ошибка - 4.6307497000722755
На итерации 1000 ошибка - 3.598782996852221e-07
На итерации 2000 ошибка - 7.395670893700428e-08
На итерации 3000 ошибка - 2.9716082815849802e-08
На итерации 4000 ошибка - 1.560325762407876e-08
На итерации 5000 ошибка - 9.479222995689099e-09
На итерации 6000 ошибка - 6.313220817519294e-09
На итерации 7000 ошибка - 4.479650956636285e-09
На итерации 8000 ошибка - 3.3292990403212777e-09
На итерации 9000 ошибка - 2.5633582396200277e-09
__________________________________________________
Итоговая ошибка 2.029559646661728e-09

Обучение заняло 2.276326894760132 сек
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Протестируем работу обученной модели. Посмотрим какой результат она выдаст на наши входные данные.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[19]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s1">'По итогу обучения модель считает:'</span><span class="p">)</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">lyters</span><span class="p">)):</span>
    <span class="n">data</span> <span class="o">=</span> <span class="n">lyters</span><span class="p">[</span><span class="n">i</span><span class="p">:</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">]</span>
    <span class="n">ans</span> <span class="o">=</span> <span class="nb">list</span><span class="p">(</span><span class="nb">map</span><span class="p">(</span><span class="nb">round</span><span class="p">,</span> <span class="n">Lyters</span><span class="o">.</span><span class="n">Predict</span><span class="p">(</span><span class="n">data</span><span class="p">)[</span><span class="mi">0</span><span class="p">]))</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">f</span><span class="s1">'</span><span class="si">{data[0]}</span><span class="s1"> - </span><span class="si">{ans}</span><span class="s1">'</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>По итогу обучения модель считает:
[1 0 1 0 1 0 1 0 1] - [0, 0, 0, 1]
[1 0 1 0 1 0 0 1 0] - [0, 0, 1, 0]
[0 1 0 0 1 0 0 1 0] - [0, 1, 0, 0]
[1 1 1 1 0 0 1 1 1] - [1, 0, 0, 0]
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>Вывод</strong><br>
Так как сумма всех значений выходного слоя после использования <strong>softmax</strong> равна 1, несложно произвести классификацию при помощи округления значений. Итоговая точность значения будет 2.029559646661728e-09 соответсвенно.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Дополнительные-задания-к-практической-работе-No-7.">Дополнительные задания к практической работе No 7.<a class="anchor-link" href="#%D0%94%D0%BE%D0%BF%D0%BE%D0%BB%D0%BD%D0%B8%D1%82%D0%B5%D0%BB%D1%8C%D0%BD%D1%8B%D0%B5-%D0%B7%D0%B0%D0%B4%D0%B0%D0%BD%D0%B8%D1%8F-%D0%BA-%D0%BF%D1%80%D0%B0%D0%BA%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%BE%D0%B9-%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B5-No-7.">¶</a></h2><h3 id="Задание-7.1.1.">Задание 7.1.1.<a class="anchor-link" href="#%D0%97%D0%B0%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5-7.1.1.">¶</a></h3><p><strong>Задание:</strong> cоздать нейронную сеть со структурой «многослойный персептрон», и
обучить ее распознаванию цифр, заданных пиксельной матрицей размером
7х5, используя 4 битный код на выходе. Произвести тестирование НС при
добавлении шума.</p>
<p>Изначально цифры будут выглядеть так:
<img src="https://raw.githubusercontent.com/sergo2048/Data-mining/main/Neural%20networks/images/0.png" alt="image0">
<img src="https://raw.githubusercontent.com/sergo2048/Data-mining/main/Neural%20networks/images/1.png" alt="image1">
<img src="https://raw.githubusercontent.com/sergo2048/Data-mining/main/Neural%20networks/images/2.png" alt="image2">
<img src="https://raw.githubusercontent.com/sergo2048/Data-mining/main/Neural%20networks/images/3.png" alt="image3">
<img src="https://raw.githubusercontent.com/sergo2048/Data-mining/main/Neural%20networks/images/4.png" alt="image4">
<img src="https://raw.githubusercontent.com/sergo2048/Data-mining/main/Neural%20networks/images/5.png" alt="image5">
<img src="https://raw.githubusercontent.com/sergo2048/Data-mining/main/Neural%20networks/images/6.png" alt="image6">
<img src="https://raw.githubusercontent.com/sergo2048/Data-mining/main/Neural%20networks/images/7.png" alt="image7">
<img src="https://raw.githubusercontent.com/sergo2048/Data-mining/main/Neural%20networks/images/8.png" alt="image8">
<img src="https://raw.githubusercontent.com/sergo2048/Data-mining/main/Neural%20networks/images/9.png" alt="image9"></p>
<p>При переносе в матричный вид за еденицу будем считать черный цвет, за 0 белый.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[20]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">numbers</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([[</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">],</span>
  
                    <span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span>

                    <span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span>

                    <span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span>

                    <span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span>
                    
                    <span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span>
  
                    <span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span>

                    <span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span>

                    <span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span>

                    <span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                     <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">]],</span> <span class="n">dtype</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">float128</span><span class="p">)</span>


<span class="n">numbers_answer</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span>  <span class="c1"># 1</span>
                           <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span>  <span class="c1"># 2</span>
                           <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span>  <span class="c1"># 3</span>
                           <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span>  <span class="c1"># 4</span>
                           <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span>  <span class="c1"># 5</span>
                           <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span>  <span class="c1"># 6</span>
                           <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span>  <span class="c1"># 7</span>
                           <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span>  <span class="c1"># 8</span>
                           <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span>  <span class="c1"># 9</span>
                           <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">]])</span> <span class="c1"># 0</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Данная нейронная сеть должны быть умнее, поэтому здесь нам не будет хватать relu функции. Поэтому в дальнейшем мы будем использовать sigmoid в качестве функции активации, а так же кросс-энтропию в качестве функции потерь.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[21]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Numbers</span> <span class="o">=</span> <span class="n">NeuralNetwork</span><span class="p">(</span><span class="n">dataset</span><span class="o">=</span><span class="n">numbers</span><span class="p">,</span>
                        <span class="n">answer</span><span class="o">=</span><span class="n">numbers_answer</span><span class="p">,</span>
                        <span class="n">neural_count</span><span class="o">=</span><span class="mi">15</span><span class="p">,</span>
                        <span class="n">iteration_count</span><span class="o">=</span><span class="mi">20000</span><span class="p">,</span>
                        <span class="n">func1</span><span class="o">=</span><span class="n">sigmoid</span><span class="p">,</span>
                        <span class="n">func2</span><span class="o">=</span><span class="n">sigmoid</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Произведем обучение модели, путем изменения весовых коэффициентов.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[22]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Numbers</span><span class="o">.</span><span class="n">Train</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>На итерации 0 ошибка - 10.502539468463692553
На итерации 2000 ошибка - 0.024845935606401343265
На итерации 4000 ошибка - 0.004947737261356947899
На итерации 6000 ошибка - 0.0015568962294123111726
На итерации 8000 ошибка - 0.0007166011025013609525
На итерации 10000 ошибка - 0.00040214141985363674558
На итерации 12000 ошибка - 0.0002494918778085184155
На итерации 14000 ошибка - 0.00016518246638597386531
На итерации 16000 ошибка - 0.00011550992393597822489
На итерации 18000 ошибка - 8.678389071532254124e-05
__________________________________________________
Итоговая ошибка 7.190813318555330623e-05

Обучение заняло 18.669897317886353 сек
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Теперь добавим некий шум к нашим исходным данным.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[23]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">numbers_mod</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([[</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">],</span>

                        <span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span>

                        <span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span>

                        <span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span>

                        <span class="p">[</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span>

                        <span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span>

                        <span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span>

                        <span class="p">[</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span>

                        <span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">],</span>

                        <span class="p">[</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span>
                         <span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">]],</span> <span class="n">dtype</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">float128</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Далее протестируем работу обученной модели, на модифицированных данных и посмотрим какой результат она выдаст.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[24]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s1">'На модифицированные данные нейронная сеть отвечает:'</span><span class="p">)</span>
<span class="n">error</span> <span class="o">=</span> <span class="mi">0</span>

<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">numbers_mod</span><span class="p">)):</span>
    <span class="n">data</span> <span class="o">=</span> <span class="n">numbers_mod</span><span class="p">[</span><span class="n">i</span><span class="p">:</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">]</span>
    <span class="n">ans</span> <span class="o">=</span> <span class="n">Numbers</span><span class="o">.</span><span class="n">Predict</span><span class="p">(</span><span class="n">data</span><span class="p">)</span>
    <span class="n">error</span> <span class="o">+=</span> <span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">((</span><span class="n">ans</span> <span class="o">-</span> <span class="n">Numbers</span><span class="o">.</span><span class="n">answer</span><span class="p">[</span><span class="n">i</span><span class="p">:</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">])</span> <span class="o">**</span> <span class="mi">2</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">f</span><span class="s1">'</span><span class="si">{ans[0]}</span><span class="s1">'</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="n">f</span><span class="s1">'Ошибка в предсказании составила - </span><span class="si">{error}</span><span class="s1">'</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>На модифицированные данные нейронная сеть отвечает:
[1.91543453e-08 6.10578145e-04 1.12144584e-04 9.99960418e-01]
[9.84597373e-01 1.30963527e-08 2.50615712e-01 4.08721141e-06]
[8.97895358e-01 2.14842656e-04 1.12384944e-02 9.99971155e-01]
[6.08719616e-06 9.99993867e-01 6.20477327e-14 7.23432953e-07]
[0.13311161 0.89523078 0.08161367 0.65970902]
[0.5 0.5 0.5 0.5]
[0.30234073 0.57035338 0.556819   0.72483873]
[9.99999955e-01 2.55151182e-08 2.38825608e-08 3.76157350e-04]
[9.98108071e-01 1.23138037e-03 8.20926681e-04 9.98081711e-01]
[9.99999955e-01 2.55423262e-08 2.39095719e-08 3.75783349e-04]
Ошибка в предсказании составила - 6.014167595076182
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Задание-7.1.3.">Задание 7.1.3.<a class="anchor-link" href="#%D0%97%D0%B0%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5-7.1.3.">¶</a></h3><p><strong>Задание:</strong> Создать нейронную сеть со структурой «многослойный персептрон», и обучить ее распознаванию четности цифр, заданных пиксельной матрицей размером 7х5. Произвести тестирование НС при добавлении шума.<br>
В переменную <strong>numbers_odd_answer</strong> записываем данные на которых будет производится обучение модели, а также берем numbers из прошлого задания.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[25]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">numbers_odd_answer</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([[</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">]])</span><span class="o">.</span><span class="n">T</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Создадим объект класса NeuralNetwork.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[26]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Numbers_odd</span> <span class="o">=</span> <span class="n">NeuralNetwork</span><span class="p">(</span><span class="n">dataset</span><span class="o">=</span><span class="n">numbers</span><span class="p">,</span>
                            <span class="n">answer</span><span class="o">=</span><span class="n">numbers_odd_answer</span><span class="p">,</span>
                            <span class="n">neural_count</span><span class="o">=</span><span class="mi">15</span><span class="p">,</span>
                            <span class="n">iteration_count</span><span class="o">=</span><span class="mi">10000</span><span class="p">,</span>
                            <span class="n">func1</span><span class="o">=</span><span class="n">sigmoid</span><span class="p">,</span>
                            <span class="n">func2</span><span class="o">=</span><span class="n">sigmoid</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Произведем обучение.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[27]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Numbers_odd</span><span class="o">.</span><span class="n">Train</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>На итерации 0 ошибка - 3.1027190134881182674
На итерации 1000 ошибка - 0.00016549573241578388573
На итерации 2000 ошибка - 4.1678749263587366035e-05
На итерации 3000 ошибка - 1.8998197187782271325e-05
На итерации 4000 ошибка - 1.0728690188161379414e-05
На итерации 5000 ошибка - 6.7779811369194759064e-06
На итерации 6000 ошибка - 4.6082142750719164525e-06
На итерации 7000 ошибка - 3.3078082907775639422e-06
На итерации 8000 ошибка - 2.477602633940519356e-06
На итерации 9000 ошибка - 1.9203997164728888466e-06
__________________________________________________
Итоговая ошибка 1.5309250264539209353e-06

Обучение заняло 9.536542654037476 сек
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Далее протестируем работу обученной модели, на модифицированных данных и посмотрим какой результат она выдаст.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[28]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s1">'По итогу обучения модель считает:'</span><span class="p">)</span>
<span class="n">error</span> <span class="o">=</span> <span class="mi">0</span>

<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">numbers_mod</span><span class="p">)):</span>
    <span class="n">data</span> <span class="o">=</span> <span class="n">numbers_mod</span><span class="p">[</span><span class="n">i</span><span class="p">:</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">]</span>
    <span class="n">ans</span> <span class="o">=</span> <span class="n">Numbers_odd</span><span class="o">.</span><span class="n">Predict</span><span class="p">(</span><span class="n">data</span><span class="p">)[</span><span class="mi">0</span><span class="p">][</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">error</span> <span class="o">+=</span> <span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">((</span><span class="n">ans</span> <span class="o">-</span> <span class="n">Numbers_odd</span><span class="o">.</span><span class="n">answer</span><span class="p">[</span><span class="n">i</span><span class="p">:</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">])</span> <span class="o">**</span> <span class="mi">2</span><span class="p">)</span>
    <span class="n">ans</span> <span class="o">=</span> <span class="s1">'Четное'</span> <span class="k">if</span> <span class="nb">round</span><span class="p">(</span><span class="n">ans</span><span class="p">)</span> <span class="k">else</span> <span class="s1">'Не четное'</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">f</span><span class="s1">'{(i + 1) % 10} - </span><span class="si">{ans}</span><span class="s1">'</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="n">f</span><span class="s1">'Ошибка в предсказании составила - </span><span class="si">{error}</span><span class="s1">'</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>По итогу обучения модель считает:
1 - Не четное
2 - Четное
3 - Не четное
4 - Четное
5 - Не четное
6 - Четное
7 - Не четное
8 - Четное
9 - Не четное
0 - Четное
Ошибка в предсказании составила - 7.898141786469726e-06
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Задание-7.1.4.">Задание 7.1.4.<a class="anchor-link" href="#%D0%97%D0%B0%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5-7.1.4.">¶</a></h3><p><strong>Задание:</strong> Создать нейронную сеть со структурой «многослойный персептрон», и обучить ее распознаванию нечетности цифр, заданных пиксельной матрицей размером 7х5. Произвести тестирование НС при добавлении шума.<br>
В этого в переменную <strong><em>numbers_even_answer</em></strong> записываем данные на которых будет производится обучение модели, а также берем numbers из прошлого задания.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[29]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">numbers_even_answer</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([[</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">]])</span><span class="o">.</span><span class="n">T</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Создадим объект класса NeuralNetwork.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[30]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Numbers_even</span> <span class="o">=</span> <span class="n">NeuralNetwork</span><span class="p">(</span><span class="n">dataset</span><span class="o">=</span><span class="n">numbers</span><span class="p">,</span>
                            <span class="n">answer</span><span class="o">=</span><span class="n">numbers_even_answer</span><span class="p">,</span>
                            <span class="n">neural_count</span><span class="o">=</span><span class="mi">15</span><span class="p">,</span>
                            <span class="n">iteration_count</span><span class="o">=</span><span class="mi">10000</span><span class="p">,</span>
                            <span class="n">func1</span><span class="o">=</span><span class="n">sigmoid</span><span class="p">,</span>
                            <span class="n">func2</span><span class="o">=</span><span class="n">sigmoid</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Произведем обучение.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[31]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Numbers_even</span><span class="o">.</span><span class="n">Train</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>На итерации 0 ошибка - 2.6470274485642733447
На итерации 1000 ошибка - 7.941719813097072918e-05
На итерации 2000 ошибка - 2.7339992701597333521e-05
На итерации 3000 ошибка - 1.4785253034511263706e-05
На итерации 4000 ошибка - 9.246420057771774955e-06
На итерации 5000 ошибка - 6.2224355519073858853e-06
На итерации 6000 ошибка - 4.39862226433083125e-06
На итерации 7000 ошибка - 3.2311548305648417518e-06
На итерации 8000 ошибка - 2.45075695545051909e-06
На итерации 9000 ошибка - 1.9103662798935275242e-06
__________________________________________________
Итоговая ошибка 1.5249789766015379178e-06

Обучение заняло 9.544184446334839 сек
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Далее протестируем работу обученной модели, на модифицированных данных и посмотрим какой результат она выдаст.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[32]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s1">'По итогу обучения модель считает:'</span><span class="p">)</span>
<span class="n">error</span> <span class="o">=</span> <span class="mi">0</span>

<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">numbers_mod</span><span class="p">)):</span>
    <span class="n">data</span> <span class="o">=</span> <span class="n">numbers_mod</span><span class="p">[</span><span class="n">i</span><span class="p">:</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">]</span>
    <span class="n">ans</span> <span class="o">=</span> <span class="n">Numbers_even</span><span class="o">.</span><span class="n">Predict</span><span class="p">(</span><span class="n">data</span><span class="p">)[</span><span class="mi">0</span><span class="p">][</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">error</span> <span class="o">+=</span> <span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">((</span><span class="n">ans</span> <span class="o">-</span> <span class="n">Numbers_even</span><span class="o">.</span><span class="n">answer</span><span class="p">[</span><span class="n">i</span><span class="p">:</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">])</span> <span class="o">**</span> <span class="mi">2</span><span class="p">)</span>
    <span class="n">ans</span> <span class="o">=</span> <span class="s1">'Не четное'</span> <span class="k">if</span> <span class="nb">round</span><span class="p">(</span><span class="n">ans</span><span class="p">)</span> <span class="k">else</span> <span class="s1">'Четное'</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">f</span><span class="s1">'{(i + 1) % 10} - </span><span class="si">{ans}</span><span class="s1">'</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="n">f</span><span class="s1">'Ошибка в предсказании составила - </span><span class="si">{error}</span><span class="s1">'</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>По итогу обучения модель считает:
1 - Не четное
2 - Четное
3 - Не четное
4 - Четное
5 - Не четное
6 - Четное
7 - Не четное
8 - Четное
9 - Не четное
0 - Четное
Ошибка в предсказании составила - 1.2265074419934752e-05
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Задание-7.1.5.">Задание 7.1.5.<a class="anchor-link" href="#%D0%97%D0%B0%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5-7.1.5.">¶</a></h3><p><strong>Задание:</strong> Создать нейронную сеть со структурой «многослойный персептрон», и обучить ее распознаванию простых чисел (от 0 до 9), заданных пиксельной матрицей размером 7х5. Произвести тестирование НС при добавлении шума.<br>
Как и до этого в переменную <strong><em>numbers_simpe_answer</em></strong> записываем данные на которых будет производится обучение модели, а также берем numbers из прошлого задания.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[33]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">numbers_simple_answer</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">]])</span><span class="o">.</span><span class="n">T</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Создадим объект класса NeuralNetwork.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[34]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Numbers_simple</span> <span class="o">=</span> <span class="n">NeuralNetwork</span><span class="p">(</span><span class="n">dataset</span><span class="o">=</span><span class="n">numbers</span><span class="p">,</span>
                               <span class="n">answer</span><span class="o">=</span><span class="n">numbers_simple_answer</span><span class="p">,</span>
                               <span class="n">neural_count</span><span class="o">=</span><span class="mi">15</span><span class="p">,</span>
                               <span class="n">iteration_count</span><span class="o">=</span><span class="mi">10000</span><span class="p">,</span>
                               <span class="n">func1</span><span class="o">=</span><span class="n">sigmoid</span><span class="p">,</span>
                               <span class="n">func2</span><span class="o">=</span><span class="n">sigmoid</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Произведем обучение.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[35]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Numbers_simple</span><span class="o">.</span><span class="n">Train</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>На итерации 0 ошибка - 3.480763845949079316
На итерации 1000 ошибка - 0.0030942048547330868886
На итерации 2000 ошибка - 0.00078816852956844107963
На итерации 3000 ошибка - 0.00034912105051482272676
На итерации 4000 ошибка - 0.00019486493288574337441
На итерации 5000 ошибка - 0.0001235686368483254216
На итерации 6000 ошибка - 8.498118368715049442e-05
На итерации 7000 ошибка - 6.182688818798110338e-05
На итерации 8000 ошибка - 4.6882165786939831442e-05
На итерации 9000 ошибка - 3.6697137679939213896e-05
__________________________________________________
Итоговая ошибка 2.946307512795766544e-05

Обучение заняло 9.505537748336792 сек
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Далее протестируем работу обученной модели, на модифицированных данных и посмотрим какой результат она выдаст.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[36]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s1">'По итогу обучения модель считает:'</span><span class="p">)</span>
<span class="n">error</span> <span class="o">=</span> <span class="mi">0</span>

<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">numbers_mod</span><span class="p">)):</span>
    <span class="n">data</span> <span class="o">=</span> <span class="n">numbers_mod</span><span class="p">[</span><span class="n">i</span><span class="p">:</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">]</span>
    <span class="n">ans</span> <span class="o">=</span> <span class="n">Numbers_simple</span><span class="o">.</span><span class="n">Predict</span><span class="p">(</span><span class="n">data</span><span class="p">)[</span><span class="mi">0</span><span class="p">][</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">error</span> <span class="o">+=</span> <span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">((</span><span class="n">ans</span> <span class="o">-</span> <span class="n">Numbers_simple</span><span class="o">.</span><span class="n">answer</span><span class="p">[</span><span class="n">i</span><span class="p">:</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">])</span> <span class="o">**</span> <span class="mi">2</span><span class="p">)</span>
    <span class="n">ans</span> <span class="o">=</span> <span class="s1">'Простое'</span> <span class="k">if</span> <span class="nb">round</span><span class="p">(</span><span class="n">ans</span><span class="p">)</span> <span class="k">else</span> <span class="s1">'Не простое'</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">f</span><span class="s1">'{(i + 1) % 10} - </span><span class="si">{ans}</span><span class="s1">'</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="n">f</span><span class="s1">'Ошибка в предсказании составила - </span><span class="si">{error}</span><span class="s1">'</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>По итогу обучения модель считает:
1 - Простое
2 - Простое
3 - Простое
4 - Не простое
5 - Простое
6 - Не простое
7 - Простое
8 - Не простое
9 - Не простое
0 - Не простое
Ошибка в предсказании составила - 0.16101737223212786
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Задание-7.1.6.">Задание 7.1.6.<a class="anchor-link" href="#%D0%97%D0%B0%D0%B4%D0%B0%D0%BD%D0%B8%D0%B5-7.1.6.">¶</a></h3><p><strong>Задание:</strong> Создать нейронную сеть со структурой «многослойный персептрон», и обучить ее распознаванию чисел,делящихся на 3 без остатка, заданных пиксельной матрицей размером 7х5. Произвести тестирование НС при добавлении шума.<br>
В переменную <strong><em>numbers_3_answer</em></strong> записываем данные на которых будет производится обучение модели, а также берем numbers из прошлого задания.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[37]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">numbers_3_answer</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([[</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">]])</span><span class="o">.</span><span class="n">T</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Создадим объект класса NeuralNetwork.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[38]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Numbers_3</span> <span class="o">=</span> <span class="n">NeuralNetwork</span><span class="p">(</span><span class="n">dataset</span><span class="o">=</span><span class="n">numbers</span><span class="p">,</span>
                          <span class="n">answer</span><span class="o">=</span><span class="n">numbers_3_answer</span><span class="p">,</span>
                          <span class="n">neural_count</span><span class="o">=</span><span class="mi">15</span><span class="p">,</span>
                          <span class="n">iteration_count</span><span class="o">=</span><span class="mi">10000</span><span class="p">,</span>
                          <span class="n">func1</span><span class="o">=</span><span class="n">sigmoid</span><span class="p">,</span>
                          <span class="n">func2</span><span class="o">=</span><span class="n">sigmoid</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Произведем обучение.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[39]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Numbers_3</span><span class="o">.</span><span class="n">Train</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>На итерации 0 ошибка - 2.5691323938386124155
На итерации 1000 ошибка - 1.1073242389110018395
На итерации 2000 ошибка - 1.0486084699479524283
На итерации 3000 ошибка - 1.030759901410942031
На итерации 4000 ошибка - 1.0226900767681845961
На итерации 5000 ошибка - 1.0180781036672698534
На итерации 6000 ошибка - 1.0150743781378202239
На итерации 7000 ошибка - 1.0129503159196844171
На итерации 8000 ошибка - 1.011361505648661539
На итерации 9000 ошибка - 1.0101238291649589962
__________________________________________________
Итоговая ошибка 1.0091306310813316501

Обучение заняло 9.17531943321228 сек
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Далее протестируем работу обученной модели, на модифицированных данных и посмотрим какой результат она выдаст.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[40]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s1">'На модифицированные данные нейронная сеть отвечает:'</span><span class="p">)</span>
<span class="n">error</span> <span class="o">=</span> <span class="mi">0</span>

<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">numbers_mod</span><span class="p">)):</span>
    <span class="n">data</span> <span class="o">=</span> <span class="n">numbers_mod</span><span class="p">[</span><span class="n">i</span><span class="p">:</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">]</span>
    <span class="n">ans</span> <span class="o">=</span> <span class="n">Numbers_3</span><span class="o">.</span><span class="n">Predict</span><span class="p">(</span><span class="n">data</span><span class="p">)[</span><span class="mi">0</span><span class="p">][</span><span class="mi">0</span><span class="p">]</span>
    <span class="n">error</span> <span class="o">+=</span> <span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">((</span><span class="n">ans</span> <span class="o">-</span> <span class="n">Numbers_3</span><span class="o">.</span><span class="n">answer</span><span class="p">[</span><span class="n">i</span><span class="p">:</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">])</span> <span class="o">**</span> <span class="mi">2</span><span class="p">)</span>
    <span class="n">ans</span> <span class="o">=</span> <span class="s1">'Делится на 3 без остатка'</span> <span class="k">if</span> <span class="nb">round</span><span class="p">(</span><span class="n">ans</span><span class="p">)</span> <span class="k">else</span> <span class="s1">'Не делится на 3 без остатка'</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">f</span><span class="s1">'{(i + 1) % 10} - </span><span class="si">{ans}</span><span class="s1">'</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="n">f</span><span class="s1">'Ошибка в предсказании составила - </span><span class="si">{error}</span><span class="s1">'</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>На модифицированные данные нейронная сеть отвечает:
1 - Не делится на 3 без остатка
2 - Не делится на 3 без остатка
3 - Делится на 3 без остатка
4 - Не делится на 3 без остатка
5 - Делится на 3 без остатка
6 - Делится на 3 без остатка
7 - Не делится на 3 без остатка
8 - Делится на 3 без остатка
9 - Не делится на 3 без остатка
0 - Делится на 3 без остатка
Ошибка в предсказании составила - 1.435167247185581
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>Вывод</strong><br>
В данных задания мы проводили тесты на модифицированных данных, соответвено точность предсказания была ниже. Это обусловленно не большой выборкой данных. Так же не стоит забывать про возможное переобучение сети.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Лабораторная-работа-7.2.">Лабораторная работа 7.2.<a class="anchor-link" href="#%D0%9B%D0%B0%D0%B1%D0%BE%D1%80%D0%B0%D1%82%D0%BE%D1%80%D0%BD%D0%B0%D1%8F-%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0-7.2.">¶</a></h1><h2 id="Искусственный-нос">Искусственный нос<a class="anchor-link" href="#%D0%98%D1%81%D0%BA%D1%83%D1%81%D1%81%D1%82%D0%B2%D0%B5%D0%BD%D0%BD%D1%8B%D0%B9-%D0%BD%D0%BE%D1%81">¶</a></h2><p><strong>Цель</strong> - разработать и исслодовать ИНС обратного распределения для искусственного носа, предназначенного для химического анализа воздушной среды.<br>
В переменные <strong><em>impurity</em></strong> и <strong><em>impurity_answer</em></strong> записываем данные на которых будет производится обучение модели.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[41]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">impurity</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([[</span><span class="mi">1</span><span class="p">,</span> <span class="mf">0.05</span><span class="p">,</span> <span class="mf">0.1</span><span class="p">,</span> <span class="mf">0.3</span><span class="p">,</span> <span class="mf">0.07</span><span class="p">,</span> <span class="mf">0.08</span><span class="p">,</span> <span class="mf">0.2</span><span class="p">,</span> <span class="mf">0.05</span><span class="p">,</span> <span class="mf">0.2</span><span class="p">,</span> <span class="mf">0.6</span><span class="p">,</span> <span class="mf">0.8</span><span class="p">],</span>
                     <span class="p">[</span><span class="mf">0.8</span><span class="p">,</span> <span class="mf">0.4</span><span class="p">,</span> <span class="mf">0.7</span><span class="p">,</span> <span class="mf">0.6</span><span class="p">,</span> <span class="mf">0.1</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mf">0.75</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.7</span><span class="p">,</span> <span class="mf">0.8</span><span class="p">],</span>
                     <span class="p">[</span><span class="mf">0.9</span><span class="p">,</span> <span class="mf">0.2</span><span class="p">,</span> <span class="mf">0.4</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.1</span><span class="p">,</span> <span class="mf">0.7</span><span class="p">,</span> <span class="mf">0.6</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.7</span><span class="p">,</span> <span class="mf">0.8</span><span class="p">],</span>
                     <span class="p">[</span><span class="mf">0.85</span><span class="p">,</span> <span class="mf">0.7</span><span class="p">,</span> <span class="mf">0.8</span><span class="p">,</span> <span class="mf">0.65</span><span class="p">,</span> <span class="mf">0.1</span><span class="p">,</span> <span class="mf">0.4</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mf">0.7</span><span class="p">,</span> <span class="mf">0.4</span><span class="p">,</span> <span class="mf">0.6</span><span class="p">,</span> <span class="mf">0.7</span><span class="p">],</span>
                     <span class="p">[</span><span class="mf">0.9</span><span class="p">,</span> <span class="mf">0.3</span><span class="p">,</span> <span class="mf">0.3</span><span class="p">,</span> <span class="mf">0.4</span><span class="p">,</span> <span class="mf">0.04</span><span class="p">,</span> <span class="mf">0.1</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.3</span><span class="p">,</span> <span class="mf">0.2</span><span class="p">,</span> <span class="mf">0.7</span><span class="p">,</span> <span class="mf">0.8</span><span class="p">],</span>
                     <span class="p">[</span><span class="mf">0.95</span><span class="p">,</span> <span class="mf">0.18</span><span class="p">,</span> <span class="mf">0.21</span><span class="p">,</span> <span class="mf">0.3</span><span class="p">,</span> <span class="mf">0.05</span><span class="p">,</span> <span class="mf">0.1</span><span class="p">,</span> <span class="mf">0.3</span><span class="p">,</span> <span class="mf">0.2</span><span class="p">,</span> <span class="mf">0.2</span><span class="p">,</span> <span class="mf">0.5</span><span class="p">,</span> <span class="mf">0.7</span><span class="p">]],</span> <span class="n">dtype</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">float128</span><span class="p">)</span>



<span class="n">impurity_answer</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span>  <span class="c1"># Нет</span>
                            <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span>  <span class="c1"># Ацетон</span>
                            <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span>  <span class="c1"># Аммиак</span>
                            <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span>  <span class="c1"># Изопропанол</span>
                            <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">],</span>  <span class="c1"># Белый "штрих"</span>
                            <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">]])</span> <span class="c1"># Уксус</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Создадим объект класса NeuralNetwork. В данное задаче как и в задаче с определением буквы для классификации лучше всего использовать функцию <strong>softmax</strong>.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[42]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Impurity</span> <span class="o">=</span> <span class="n">NeuralNetwork</span><span class="p">(</span><span class="n">dataset</span><span class="o">=</span><span class="n">impurity</span><span class="p">,</span>
                         <span class="n">answer</span><span class="o">=</span><span class="n">impurity_answer</span><span class="p">,</span>
                         <span class="n">neural_count</span><span class="o">=</span><span class="mi">15</span><span class="p">,</span>
                         <span class="n">iteration_count</span><span class="o">=</span><span class="mi">10000</span><span class="p">,</span>
                         <span class="n">func2</span><span class="o">=</span><span class="n">softmax</span><span class="p">)</span>
<span class="n">Impurity</span><span class="o">.</span><span class="n">GenerateWeights</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Произведем обучение.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[43]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">Impurity</span><span class="o">.</span><span class="n">Train</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>На итерации 0 ошибка - 8.246700422670458259
На итерации 1000 ошибка - 2.5970464507474085737e-05
На итерации 2000 ошибка - 4.7086903420172897942e-06
На итерации 3000 ошибка - 1.7900761639016368543e-06
На итерации 4000 ошибка - 9.1033416436891036594e-07
На итерации 5000 ошибка - 5.4138258031273930627e-07
На итерации 6000 ошибка - 3.550754661487755993e-07
На итерации 7000 ошибка - 2.4888190437295416145e-07
На итерации 8000 ошибка - 1.8308380681743554993e-07
На итерации 9000 ошибка - 1.3986387121062363635e-07
__________________________________________________
Итоговая ошибка 1.0997339600765824284e-07

Обучение заняло 3.4211952686309814 сек
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Протестируем работу обученной сети.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[44]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s1">'По итогу обучения модель считает:'</span><span class="p">)</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">impurity</span><span class="p">)):</span>
    <span class="n">data</span> <span class="o">=</span> <span class="n">impurity</span><span class="p">[</span><span class="n">i</span><span class="p">:</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">]</span>
    <span class="n">ans</span> <span class="o">=</span> <span class="nb">list</span><span class="p">(</span><span class="nb">map</span><span class="p">(</span><span class="nb">round</span><span class="p">,</span> <span class="n">Impurity</span><span class="o">.</span><span class="n">Predict</span><span class="p">(</span><span class="n">data</span><span class="p">)[</span><span class="mi">0</span><span class="p">]))</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">f</span><span class="s1">'</span><span class="si">{ans}</span><span class="s1"> - </span><span class="si">{data[0]}</span><span class="s1">'</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>По итогу обучения модель считает:
[0, 0, 0, 0, 0, 1] - [1.   0.05 0.1  0.3  0.07 0.08 0.2  0.05 0.2  0.6  0.8 ]
[0, 0, 0, 0, 1, 0] - [0.8  0.4  0.7  0.6  0.1  0.5  1.   0.75 0.5  0.7  0.8 ]
[0, 0, 0, 1, 0, 0] - [0.9 0.2 0.4 0.5 0.1 0.7 0.6 0.5 0.5 0.7 0.8]
[0, 0, 1, 0, 0, 0] - [0.85 0.7  0.8  0.65 0.1  0.4  1.   0.7  0.4  0.6  0.7 ]
[0, 1, 0, 0, 0, 0] - [0.9  0.3  0.3  0.4  0.04 0.1  0.5  0.3  0.2  0.7  0.8 ]
[1, 0, 0, 0, 0, 0] - [0.95 0.18 0.21 0.3  0.05 0.1  0.3  0.2  0.2  0.5  0.7 ]
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><strong>Вывод</strong><br>
Так как сумма всех значений выходного слоя после использования <strong>softmax</strong> равна 1, несложно произвести классификацию при помощи округления значений. Итоговая точность значения будет 1.0997339600765824284e-07 соответсвенно.</p>

</div>
</div>
</div>
    </div>
  </div>


 



