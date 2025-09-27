texts = [
    # Original entries (36)
    "Patient with recurrent UTI, E. coli positive, resistant to ciprofloxacin; started on ceftriaxone.",
    "Blood culture shows MRSA bacteremia; vancomycin initiated, contact precautions in place.",
    "Wound swab positive for Pseudomonas aeruginosa, multidrug-resistant; no isolation protocol noted.",
    "Patient with ventilator-associated pneumonia; culture shows Klebsiella pneumoniae, ESBL-positive.",
    "Sputum culture negative after 5 days of levofloxacin; hand hygiene compliance documented.",
    "Catheter-associated UTI suspected; urine culture pending, no sterile technique mentioned.",
    "Patient with fever and cellulitis; MRSA confirmed, isolation protocol initiated.",
    "Bloodstream infection with VRE; linezolid started, poor hand hygiene compliance noted.",
    "Post-surgical wound infection; culture shows susceptible Staphylococcus aureus, contact precautions adequate.",
    "Patient with recurrent sepsis; cultures show carbapenem-resistant Enterobacteriaceae.",
    "Antibiotic history shows repeated amoxicillin failure; resistance suspected, no isolation noted.",
    "C. difficile infection confirmed; metronidazole started, contact precautions in place.",
    "Patient with pneumonia; sputum culture shows pan-sensitive Streptococcus pneumoniae.",
    "Central line infection suspected; culture positive for Acinetobacter baumannii, multidrug-resistant.",
    "UTI with Proteus mirabilis, susceptible to nitrofurantoin; no IPC measures documented.",
    "Surgical site infection; swab shows MRSA, isolation not implemented.",
    "Patient with chronic wound; culture negative, no antibiotic resistance noted, IPC adequate.",
    "Septic arthritis with resistant Enterococcus; vancomycin used, IPC measures adequate.",
    "Fever in neutropenic patient; blood culture shows ESBL E. coli, no contact precautions.",
    "Post-operative fever; culture shows susceptible E. coli, sterile technique documented.",
    "Patient with recurrent cellulitis; MRSA suspected, no hand hygiene compliance noted.",
    "Sputum culture shows carbapenem-resistant Pseudomonas; isolation protocol initiated.",
    "UTI with Klebsiella, resistant to ceftriaxone; meropenem started, IPC inadequate.",
    "Blood culture negative after 7 days of antibiotics; contact precautions maintained.",
    "Wound infection with VRE; linezolid initiated, no isolation protocol mentioned.",
    "Patient with fever and cough; culture shows susceptible Haemophilus influenzae.",
    "Post-surgical infection; swab positive for multidrug-resistant Acinetobacter, no IPC.",
    "C. difficile colitis confirmed; vancomycin oral, hand hygiene emphasized.",
    "Patient with catheter-related infection; culture shows resistant Klebsiella, no precautions.",
    "Pneumonia with pan-sensitive S. pneumoniae; ceftriaxone effective, IPC adequate.",
    "Septicemia with MRSA; vancomycin started, contact precautions not documented.",
    "Chronic osteomyelitis; culture shows multidrug-resistant S. aureus, isolation in place.",
    "Patient with UTI; culture negative, no antibiotic resistance noted, IPC adequate.",
    "Bloodstream infection with ESBL-positive E. coli; meropenem used, no hand hygiene.",
    "Post-operative wound infection; culture shows susceptible Enterococcus, IPC documented.",
    "Patient with no signs of infection; routine post-op care, no antibiotics or IPC measures required.",

    # Previous expansion: Klebsiella pneumoniae (10)
    "ICU patient with ventilator-associated pneumonia; culture shows KPC-producing Klebsiella pneumoniae; ceftazidime-avibactam started, contact precautions in place.",
    "Catheter-associated UTI in elderly patient; culture positive for NDM-1 Klebsiella pneumoniae; colistin initiated, no hand hygiene compliance noted.",
    "Bloodstream infection in neutropenic patient; culture shows ESBL-positive Klebsiella pneumoniae; meropenem started, isolation protocol inadequate.",
    "Surgical site infection post-abdominal surgery; swab shows OXA-48 Klebsiella pneumoniae; tigecycline used, contact precautions implemented.",
    "Community-acquired UTI; culture shows pan-sensitive Klebsiella pneumoniae; cefuroxime initiated, no IPC measures documented.",
    "Septic arthritis in diabetic patient; culture positive for multidrug-resistant Klebsiella pneumoniae; ertapenem started, IPC adequate.",
    "Neonatal sepsis; blood culture shows susceptible Klebsiella pneumoniae; cefotaxime initiated, contact precautions in place.",
    "Intra-abdominal abscess; culture shows carbapenem-resistant Klebsiella pneumoniae; combination therapy with meropenem and colistin, no isolation noted.",
    "Pneumonia in long-term care resident; sputum culture shows ESBL-positive Klebsiella pneumoniae; piperacillin-tazobactam started, hand hygiene compliance documented.",
    "Suspected meningitis in ICU patient; CSF culture negative for Klebsiella pneumoniae; ceftriaxone started empirically, droplet precautions maintained.",

    # Previous expansion: Acinetobacter baumannii (10)
    "Ventilator-associated pneumonia in ICU; culture shows pan-resistant Acinetobacter baumannii; colistin and tigecycline started, contact precautions in place.",
    "Central line-associated bloodstream infection; culture positive for MDR Acinetobacter baumannii; sulbactam initiated, no isolation protocol noted.",
    "Chronic wound infection in diabetic patient; swab shows carbapenem-resistant Acinetobacter baumannii; minocycline used, IPC inadequate.",
    "Post-surgical wound infection; culture shows susceptible Acinetobacter baumannii; ceftazidime started, contact precautions adequate.",
    "Community-acquired pneumonia in outpatient; culture shows MDR Acinetobacter baumannii; levofloxacin initiated, no IPC measures documented.",
    "Septicemia in burn patient; blood culture positive for multidrug-resistant Acinetobacter baumannii; polymyxin B started, isolation protocol initiated.",
    "Skin and soft tissue infection post-trauma; culture shows carbapenem-resistant Acinetobacter baumannii; combination therapy with colistin and rifampin, no hand hygiene compliance.",
    "Meningitis in neurosurgery patient; CSF culture positive for MDR Acinetobacter baumannii; meropenem and colistin started, contact precautions in place.",
    "Suspected catheter-related infection; culture negative for Acinetobacter baumannii; antibiotics held, sterile technique documented.",
    "Osteomyelitis in orthopedic patient; culture shows multidrug-resistant Acinetobacter baumannii; tigecycline initiated, IPC adequate.",

    # Previous expansion: Pseudomonas aeruginosa (10)
    "Cystic fibrosis patient with chronic lung infection; sputum culture shows multidrug-resistant Pseudomonas aeruginosa; ceftolozane-tazobactam started, contact precautions in place.",
    "Catheter-associated UTI in ICU patient; culture positive for carbapenem-resistant Pseudomonas aeruginosa; colistin initiated, no hand hygiene compliance noted.",
    "Bloodstream infection in burn patient; culture shows VIM-producing Pseudomonas aeruginosa; aztreonam and colistin started, isolation protocol inadequate.",
    "Ventilator-associated pneumonia; culture positive for pan-sensitive Pseudomonas aeruginosa; piperacillin-tazobactam initiated, IPC adequate.",
    "Chronic wound infection in diabetic patient; swab shows MDR Pseudomonas aeruginosa; tobramycin inhalation therapy started, no isolation protocol noted.",
    "Septicemia in neutropenic patient; blood culture positive for carbapenem-resistant Pseudomonas aeruginosa; ceftazidime and colistin started, contact precautions implemented.",
    "Otitis externa in outpatient; culture shows susceptible Pseudomonas aeruginosa; ciprofloxacin ear drops initiated, no IPC measures required.",
    "Post-surgical wound infection; culture shows multidrug-resistant Pseudomonas aeruginosa; meropenem started, hand hygiene compliance documented.",
    "Corneal ulcer in contact lens wearer; culture positive for MDR Pseudomonas aeruginosa; levofloxacin eye drops started, no isolation protocol noted.",
    "Suspected ventilator-associated pneumonia; sputum culture negative for Pseudomonas aeruginosa; antibiotics discontinued, contact precautions maintained.",

    # Previous expansion: Sample note terms (3)
    "Fever and cough post-hospitalization, treated with ciprofloxacin, no isolation protocols.",
    "Recurrent infections, recent ceftriaxone use, inadequate hand hygiene noted.",
    "Pneumonia post-antibiotic therapy, no contact precautions in place.",

    # New examples for P1000 note and class balance (20)
    # amr_high (5): Focus on ciprofloxacin, ceftriaxone, and resistance
    "Fever and productive cough post-hospitalization; recent ciprofloxacin use, suspected resistance, no isolation noted.",
    "Patient with wound infection; ceftriaxone 1g IV failed, possible multidrug resistance, no IPC measures.",
    "Recurrent UTI post-ciprofloxacin therapy; culture pending, suspected antibiotic resistance, no precautions.",
    "Post-surgical infection; ceftriaxone ineffective, possible MRSA, hand hygiene compliance poor.",
    "Sepsis with fever and fatigue; recent ciprofloxacin and ceftriaxone use, resistance suspected, no isolation.",

    # amr_low (4): Susceptible infections
    "Pneumonia with fever and cough; culture shows susceptible Streptococcus pneumoniae, ceftriaxone effective, IPC adequate.",
    "UTI post-hospitalization; culture shows sensitive E. coli, nitrofurantoin started, contact precautions in place.",
    "Wound infection with cough; culture negative after cefazolin, no resistance, hand hygiene documented.",
    "Fever and fatigue; culture shows pan-sensitive Haemophilus influenzae, azithromycin effective, IPC maintained.",

    # amr_none (4): No infection or resistance
    "Fever resolved post-hospitalization; culture negative, no antibiotics started, no IPC needed.",
    "Patient with fatigue and cough; no infection confirmed, culture negative, routine care continued.",
    "Post-surgical check; no fever or infection, culture negative, no antibiotics required.",
    "Outpatient with cough; culture pending, no resistance suspected, no precautions needed.",

    # ipc_adequate (3): Strong IPC measures
    "Pneumonia post-ciprofloxacin; culture shows susceptible S. pneumoniae, strict contact precautions and hand hygiene enforced.",
    "Wound infection with ceftriaxone; culture positive, isolation protocol and sterile technique documented.",
    "Sepsis with fever; antibiotics started, contact precautions and hand hygiene compliance maintained.",

    # ipc_inadequate (2): Lack of IPC
    "Fever and cough post-ciprofloxacin therapy; culture positive, no isolation or hand hygiene noted.",
    "Post-surgical wound infection; ceftriaxone started, no IPC measures or precautions documented.",

    # ipc_none (2): No IPC needed
    "Patient with fatigue; no infection noted, culture negative, no antibiotics or IPC measures required.",
    "Post-hospitalization follow-up; no fever or cough, routine care, no IPC needed."
]