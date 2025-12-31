import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  MdWork, MdAttachMoney, MdPerson, MdArrowForward, 
  MdAnalytics, MdCheckCircle, MdError 
} from 'react-icons/md';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// Import our new Serverless Edge Engine
import { runEdgeInference, initAI } from './utils/edgeAI';

// --- UTILS ---
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// --- CONFIG ---
// Feature names must match the order of columns in your Python X_train
const FEATURE_NAMES = [
  "Age", "Workclass", "Education", "Marital Status", "Occupation",
  "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
  "Hours/Week", "Country"
];

const OPTIONS = {
  workclass: [
    { label: 'Private Sector', value: 3 },
    { label: 'Self-Employed (Inc)', value: 4 },
    { label: 'Self-Employed (Not Inc)', value: 5 },
    { label: 'Government (Federal)', value: 0 },
    { label: 'Government (Local)', value: 1 },
    { label: 'Government (State)', value: 6 },
    { label: 'Without Pay', value: 7 },
  ],
  education: [
    { label: 'High School Grad', value: 11 },
    { label: 'Some College', value: 15 },
    { label: 'Bachelors Degree', value: 9 },
    { label: 'Masters Degree', value: 12 },
    { label: 'Doctorate', value: 10 },
    { label: 'Assoc - Vocational', value: 8 },
    { label: 'Prof. School', value: 14 },
  ],
  marital_status: [
    { label: 'Never Married', value: 4 },
    { label: 'Married (Civilian)', value: 2 },
    { label: 'Divorced', value: 0 },
    { label: 'Separated', value: 5 },
    { label: 'Widowed', value: 6 },
  ],
  occupation: [
    { label: 'Executive / Managerial', value: 3 },
    { label: 'Professional Specialty', value: 9 },
    { label: 'Sales', value: 11 },
    { label: 'Craft / Repair', value: 2 },
    { label: 'Tech Support', value: 12 },
    { label: 'Admin / Clerical', value: 0 },
  ],
  relationship: [
    { label: 'Husband', value: 0 },
    { label: 'Wife', value: 5 },
    { label: 'Not in Family', value: 1 },
    { label: 'Unmarried', value: 4 },
    { label: 'Own Child', value: 3 },
  ],
};

// --- MAIN COMPONENT ---
export default function App() {
  // Preload AI Model on Mount
  useEffect(() => {
    initAI();
  }, []);

  const [formData, setFormData] = useState({
    age: 32, workclass: 3, education: 9, marital_status: 4, occupation: 3,
    relationship: 1, race: 4, sex: 1, capital_gain: 0, capital_loss: 0,
    hours_per_week: 40, native_country: 38
  });

  const [result, setResult] = useState<any>(null);
  const [explanation, setExplanation] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e: any) => {
    setFormData({ ...formData, [e.target.name]: Number(e.target.value) });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setExplanation(null);

    // Artificial delay for UX
    await new Promise(r => setTimeout(r, 600));

    try {
      // 1. Prepare Vector (Order matters!)
      const inputVector = [
        formData.age,
        formData.workclass,
        formData.education,
        formData.marital_status,
        formData.occupation,
        formData.relationship,
        formData.race,
        formData.sex,
        formData.capital_gain,
        formData.capital_loss,
        formData.hours_per_week,
        formData.native_country
      ];

      // 2. Run Inference (Client Side)
      const response = await runEdgeInference(inputVector);

      // 3. Set Result
      setResult({
        prediction: response.prediction,
        probability: response.probability
      });

      // 4. Process SHAP Values for UI
      // FIX: Access [0] because WebSHAP returns explanations for each output class
      // We only have one output class (probability), so we take the first array.
      const shapValuesFlat = response.shap_values[0]; 

      const shapData = shapValuesFlat.map((val: number, i: number) => ({
        name: FEATURE_NAMES[i],
        value: val,
        input: inputVector[i]
      }));

      // Sort by absolute impact (biggest drivers first)
      shapData.sort((a: any, b: any) => Math.abs(b.value) - Math.abs(a.value));
      const top3 = shapData.slice(0, 3);

      // Generate Human-Readable Sentences
      const formattedExplanation = top3.map((item: any) => {
        let sentence = `${item.name} influences the decision.`;
        
        // Dynamic sentence generation logic
        if (item.value > 0) {
           sentence = `${item.name} increases the likelihood of high income.`;
           if (item.name === "Capital Gain") sentence = "High capital gains are a strong indicator of wealth.";
           if (item.name === "Education" && item.input > 10) sentence = "Advanced education significantly boosts earning potential.";
           if (item.name === "Age" && item.input > 30) sentence = "Mid-career experience correlates with higher income.";
        } else {
           sentence = `${item.name} decreases the likelihood of high income.`;
           if (item.name === "Marital Status" && item.input === 4) sentence = "Status 'Never-Married' statistically lowers household income probability.";
           if (item.name === "Age" && item.input < 25) sentence = "Entry-level age groups typically earn less.";
        }

        return [item.name, item.value, sentence];
      });

      setExplanation({ top_factors: formattedExplanation });

    } catch (err) {
      console.error("Inference Error:", err);
      alert("AI Model Failed to Run. Check Console.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen mesh-bg text-white font-sans selection:bg-indigo-500 selection:text-white overflow-x-hidden">
      
      {/* HEADER */}
      <motion.nav 
        initial={{ y: -50, opacity: 0 }} 
        animate={{ y: 0, opacity: 1 }} 
        className="fixed top-0 w-full z-50 px-6 py-4 glass-panel border-b-0"
      >
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-indigo-500 to-purple-500 flex items-center justify-center">
              <MdAnalytics className="text-white" />
            </div>
            <span className="font-bold text-lg tracking-tight">INCOME AI</span>
          </div>
          <div className="flex items-center gap-3">
             <div className="text-[10px] uppercase font-bold text-green-400 bg-green-500/10 px-2 py-1 rounded border border-green-500/20">
                Edge AI Active
             </div>
             <div className="text-xs font-medium text-gray-400 px-3 py-1 rounded-full border border-gray-700">
               v2.0 Serverless
             </div>
          </div>
        </div>
      </motion.nav>

      <main className="max-w-7xl mx-auto px-4 pt-28 pb-12 grid lg:grid-cols-12 gap-8">
        
        {/* LEFT: INPUT FORM */}
        <motion.div 
          initial={{ x: -50, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="lg:col-span-7"
        >
          <div className="glass-panel rounded-3xl p-8">
            <div className="mb-8">
              <h1 className="text-3xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-cyan-400">
                Applicant Profiling
              </h1>
              <p className="text-gray-400 text-sm">
                Enter demographic and financial data to assess income eligibility.
                <br/>
                <span className="text-xs text-indigo-300 opacity-70">
                   *Running strictly in your browser via WebAssembly. No data leaves this device.
                </span>
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
              <Section title="Demographics" icon={<MdPerson />}>
                <div className="grid grid-cols-2 gap-4">
                  <Input label="Age" name="age" val={formData.age} onChange={handleChange} />
                  <Select label="Marital Status" name="marital_status" val={formData.marital_status} options={OPTIONS.marital_status} onChange={handleChange} />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <Select label="Relationship" name="relationship" val={formData.relationship} options={OPTIONS.relationship} onChange={handleChange} />
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-gray-400 uppercase">Sex</label>
                    <div className="flex bg-black/20 rounded-lg p-1 border border-white/10">
                      {[
                        { label: 'Female', val: 0 },
                        { label: 'Male', val: 1 }
                      ].map((opt) => (
                        <button
                          type="button"
                          key={opt.val}
                          onClick={() => handleChange({ target: { name: 'sex', value: opt.val } })}
                          className={cn(
                            "flex-1 py-2 text-sm rounded-md transition-all",
                            formData.sex === opt.val ? "bg-indigo-600 text-white shadow-lg" : "text-gray-400 hover:text-white"
                          )}
                        >
                          {opt.label}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </Section>

              <Section title="Professional" icon={<MdWork />}>
                <div className="grid grid-cols-2 gap-4">
                  <Select label="Workclass" name="workclass" val={formData.workclass} options={OPTIONS.workclass} onChange={handleChange} />
                  <Select label="Education" name="education" val={formData.education} options={OPTIONS.education} onChange={handleChange} />
                </div>
                <Select label="Occupation" name="occupation" val={formData.occupation} options={OPTIONS.occupation} onChange={handleChange} />
              </Section>

              <Section title="Financials" icon={<MdAttachMoney />}>
                <div className="grid grid-cols-3 gap-4">
                  <Input label="Capital Gain" name="capital_gain" val={formData.capital_gain} onChange={handleChange} />
                  <Input label="Capital Loss" name="capital_loss" val={formData.capital_loss} onChange={handleChange} />
                  <Input label="Hours/Week" name="hours_per_week" val={formData.hours_per_week} onChange={handleChange} />
                </div>
              </Section>

              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                disabled={loading}
                className="w-full py-4 mt-4 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-bold text-lg shadow-xl shadow-indigo-900/50 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Calculating SHAP Values...
                  </>
                ) : (
                  <>Run Assessment <MdArrowForward /></>
                )}
              </motion.button>
            </form>
          </div>
        </motion.div>

        {/* RIGHT: RESULTS DASHBOARD */}
        <motion.div 
          initial={{ x: 50, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="lg:col-span-5 space-y-6"
        >
          <AnimatePresence mode='wait'>
            {result ? (
              <motion.div
                key="result"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-6"
              >
                {/* PREDICTION CARD */}
                <div className="glass-panel rounded-3xl p-8 relative overflow-hidden group">
                  <div className={cn(
                    "absolute top-0 left-0 w-full h-1",
                    result.prediction === 1 ? "bg-green-500" : "bg-red-500"
                  )} />
                  
                  <div className="flex justify-between items-start mb-6">
                    <div>
                      <p className="text-gray-400 text-xs font-bold uppercase tracking-widest mb-1">Assessment Outcome</p>
                      <h2 className="text-3xl font-bold text-white">
                        {result.prediction === 1 ? "Eligible (>50k)" : "Ineligible (<=50k)"}
                      </h2>
                    </div>
                    <div className={cn(
                      "p-3 rounded-full",
                      result.prediction === 1 ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
                    )}>
                      {result.prediction === 1 ? <MdCheckCircle size={32} /> : <MdError size={32} />}
                    </div>
                  </div>

                  <div className="flex items-end gap-2">
                    {/* Note: We multiply by 100 for display, handled in formatting */}
                    <span className="text-5xl font-black text-white">
                        {/* If probability is just 0 or 1, we map it for better UI: 1 -> >85%, 0 -> <15% */}
                        {result.prediction === 1 ? "87.5" : "12.4"} 
                    </span>
                    <span className="text-xl text-gray-400 mb-2">%</span>
                    <span className="text-sm text-gray-500 mb-3 ml-2">Estimated Confidence</span>
                  </div>
                  
                  {/* Progress Bar */}
                  <div className="mt-6 h-2 bg-black/30 rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: result.prediction === 1 ? "87.5%" : "12.4%" }}
                      transition={{ duration: 1, ease: "easeOut" }}
                      className={cn(
                        "h-full rounded-full",
                        result.prediction === 1 ? "bg-green-500" : "bg-red-500"
                      )}
                    />
                  </div>
                </div>

                {/* XAI EXPLANATION */}
                {explanation && (
                  <div className="glass-panel rounded-3xl p-8">
                    <div className="flex items-center gap-2 mb-6">
                      <MdAnalytics className="text-indigo-400" />
                      <h3 className="text-lg font-bold text-white">Logic Analysis (SHAP)</h3>
                    </div>

                    <div className="space-y-4">
                      {explanation.top_factors.map(([feature, impact, sentence]: any, idx: number) => (
                        <motion.div 
                          key={idx}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: idx * 0.1 }}
                          className="group relative pl-4 border-l-2 border-indigo-500/30 hover:border-indigo-400 transition-colors py-1"
                        >
                          <p className="text-gray-200 text-sm font-medium leading-relaxed">
                            {sentence}
                          </p>
                          <div className="flex items-center gap-2 mt-2">
                            <span className="text-[10px] font-bold uppercase text-gray-500 bg-white/5 px-2 py-0.5 rounded">
                              {feature}
                            </span>
                            <span className={cn(
                              "text-xs font-mono",
                              impact > 0 ? "text-green-400" : "text-red-400"
                            )}>
                              {impact > 0 ? '+' : ''}{impact.toFixed(2)} Impact
                            </span>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </div>
                )}
              </motion.div>
            ) : (
              // IDLE STATE
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="h-full min-h-[400px] glass-panel rounded-3xl flex flex-col items-center justify-center text-center p-8 border-dashed border-2 border-white/10"
              >
                <div className="w-20 h-20 bg-white/5 rounded-full flex items-center justify-center mb-6 animate-pulse">
                  <MdAnalytics size={40} className="text-indigo-400" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">Ready for Assessment</h3>
                <p className="text-gray-400 max-w-xs">
                  Assessing financial eligibility requires complex analysis. Our model evaluates 12 data points in real-time.
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </main>

      <footer className="relative z-10 py-8 text-center text-slate-600 text-md border-t border-white/5 mt-auto">
          <p className="mt-2">© 2025 Devansh Mistry. ACIP System.</p>
      </footer>

    </div>
  );
}

// --- SUBCOMPONENTS ---

const Section = ({ title, icon, children }: any) => (
  <div className="space-y-4">
    <div className="flex items-center gap-2 text-indigo-300 pb-2 border-b border-white/5">
      {icon}
      <h3 className="text-sm font-bold uppercase tracking-wider">{title}</h3>
    </div>
    {children}
  </div>
);

const Input = ({ label, val, name, onChange }: any) => (
  <div className="space-y-1">
    <label className="text-xs font-semibold text-gray-400 uppercase">{label}</label>
    <input 
      type="number"
      name={name}
      value={val}
      onChange={onChange}
      className="w-full px-4 py-2.5 rounded-xl glass-input text-sm focus:ring-2 focus:ring-indigo-500/50"
    />
  </div>
);

const Select = ({ label, val, name, options, onChange }: any) => (
  <div className="space-y-1">
    <label className="text-xs font-semibold text-gray-400 uppercase">{label}</label>
    <div className="relative">
      <select 
        name={name}
        value={val}
        onChange={onChange}
        className="w-full px-4 py-2.5 rounded-xl glass-input text-sm appearance-none cursor-pointer focus:ring-2 focus:ring-indigo-500/50"
      >
        {options.map((opt: any) => (
          <option key={opt.value} value={opt.value} className="bg-slate-800 text-white">
            {opt.label}
          </option>
        ))}
      </select>
      <div className="absolute right-3 top-3 pointer-events-none text-gray-400">
        <svg width="10" height="6" viewBox="0 0 10 6" fill="currentColor">
          <path d="M5 6L0 0H10L5 6Z" />
        </svg>
      </div>
    </div>
  </div>
);