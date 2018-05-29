# Estimator Spec:
#
# Inputs:
#
# W - Word matrix, with each word in its own column and each vote in its own row
# Y - Vote matrix, with each voter it his/her own column, and each vote in its own row
#
# Column names in W should be the words themselves, Column names in the Vote matrix should 
# be the voters names.
#
# LDA.topics - Number of desired dimensions, determines the LDA topic number and the resulting 
# number of estimated issue-specific ideal points for each voter.  If a vector, the model 
# is fit for all numbers of topics sequentially, and then the model with the number of topics
# that yields the best predictions for roll-call votes is reported.
#
# LDA.alpha - Dirichlet "clumpiness" parameter for LDA (default 50/topics)

# LDA.beta - Dirichlet "clumpiness" parameter for LDA (default 0.01)

# IRT.discrim.prec - IRT discrimination parameter prior precision (default 0.25)

# IRT.polarity - Absolute value is the index of the constrained voter, 
# sign is polarity of constraint
#
# burn.iter - Number of discarded initial iterations (default = 1000)
#
# mcmc.iter - Number of recorded iterations (default = 1000)
#
# save.word.chain - Save full chain for word assignments to each topic (default = FALSE)  
# If FALSE, just save running mean.
#
# save.topic.chain - Save full chain for topic assignments for each vote (default = FALSE)  
# If FALSE, just save running mean.
#
# save.ideal.chain - Save full chain for topic-specific ideal points (default = TRUE)  
# If FALSE, just save running mean.
#
# Output:
#
# Saved object with posterior means (and mcmc chains if selected) for word assignments 
# to topic, vote assignments to topic, and ideal points for each topic.  
#
# Separate display function should take that output object and display distinctive words 
# for a topic with sorted list of voter names on that dimension.


library(Rcpp)
library(RcppArmadillo)	
library(inline)


LDAIRTinc = '

int rcat(Rcpp::NumericVector pvec){
double total = sum(pvec);
double udraw = Rf_runif(0,total);
double cumsum = pvec[0];
int draw = 0;
while (cumsum < udraw) {
cumsum = cumsum + pvec[draw+1];
draw ++;
}
return draw;  
}


//// CHECK THE REJECTION PROBABILITY EXPRESSION HERE!
double rnegtruncnorm(double mean, double sd){
double pi = 3.14159265358979323846264338327950288419716939937510;
double proposal;
double acceptprob;
double acceptdraw;
if (mean < 0){
do { 
proposal = Rf_rnorm(mean,sd);
} while (proposal > 0);
return proposal; 			
} else {
do { 
proposal = log(Rf_runif(0,1))*sqrt(2.0*pi*sd*sd);
acceptprob = exp((sqrt(2.0/pi)*sd*proposal)/((proposal - mean)*(proposal - mean)));
acceptdraw = Rf_runif(0,1);
} while (acceptdraw > acceptprob);
return proposal;   	
}
}



'

LDAIRTsrc <- '

Rcpp::RNGScope scope;

Rcpp::IntegerVector WMD(WordMatrixDocs); 
Rcpp::IntegerVector WMT(WordMatrixTerms); 
Rcpp::IntegerVector WMC(WordMatrixCounts); 
Rcpp::IntegerMatrix Y(VoteMatrix);  
int ntopics = as<int>(LDAtopics); 
int nterms = as<int>(LDAterms);
int nwords = as<int>(LDAwords);
int ndocterms = as<int>(LDAdocterms); 
double alpha = as<double>(LDAalpha);
double beta = as<double>(LDAbeta);
double prec = as<double>(IRTprec);
int polarity = as<int>(IRTpolarity);
Rcpp::LogicalVector store(SaveChains);
int burn = as<int>(BurntIterations);
int mcmc = as<int>(SavedIterations);
int nvotes = Y.nrow(); 
int nvoters = Y.ncol(); 

Rprintf("%d voters\\n",nvoters);
Rprintf("%d votes\\n",nvotes);
Rprintf("%d terms\\n",nterms);
Rprintf("%d total words\\n",nwords);

// SET UP CHAIN STORAGE OBJECTS

int wordchainstorageloc = 0;
int wordchainlength = nterms*ntopics;
if (store[0]) wordchainlength = wordchainlength*mcmc;
Rcpp::NumericVector wordchain(wordchainlength);

int topicchainstorageloc = 0;		
int topicchainlength = ntopics*nvotes;			
if (store[1]) topicchainlength = topicchainlength*mcmc;
Rcpp::NumericVector topicchain(topicchainlength);

int idealchainstorageloc = 0;	
int idealchainlength = nvoters*ntopics;
if (store[2]) idealchainlength = idealchainlength*mcmc;
Rcpp::NumericVector idealchain(idealchainlength);

int discrimchainstorageloc = 0;			
int discrimchainlength = nvotes*2;
if (store[3]) discrimchainlength = discrimchainlength*mcmc;
Rcpp::NumericVector discrimchain(discrimchainlength);

Rcpp::NumericVector rhochain(mcmc); // chain for rho

Rcpp::NumericVector llldachain(mcmc); // log-likelihood chain for LDA model
Rcpp::NumericVector llirtchain(mcmc); // log-likelihood chain for IRT model

// SET UP STATE VARIABLES	

Rcpp::NumericMatrix LDAnmz(nvotes,ntopics);						// working topic counts for each vote
Rcpp::IntegerVector LDAnm(nvotes);								// total number of terms in each document
Rcpp::NumericMatrix LDAnzt(ntopics,nterms);						// working term use by topic
Rcpp::IntegerVector LDAnz(ntopics);								// total number of term appearances for each topic
Rcpp::IntegerVector LDAzmn(nwords);								// working topic assignment for each word
Rcpp::NumericMatrix LDAphi(ntopics,nterms);						// multinomial parameters for term probabilities within topics

Rcpp::NumericMatrix LDAIRTlambda(nvotes,ntopics);				// working topic weights for each vote

Rcpp::NumericMatrix IRTystar(nvotes,nvoters);					// working latent utilities for each voter vote
Rcpp::NumericMatrix IRTtheta(nvoters,ntopics); 					// working voter ideal points for each topic
arma::mat IRTthetaarma(nvoters,ntopics);
Rcpp::NumericMatrix IRTbeta(nvotes,2);							// working discrimination parameters for each vote

// INITIALIZE WORKING VARIABLES	FOR LDA

double pvecdenom = 0;
double phidenom = 0;
double lambdadenom = 0;

double LDAll = 0;

// INITIALIZE WORKING VARIABLES	FOR IRT

double muim = 0;
Rcpp::NumericVector drawtemp(1);

arma::mat Xs(nvoters,2);
arma::mat Ys(nvoters,1);
arma::mat Tinvs(2,2);
Tinvs(0,0) = prec;
Tinvs(0,1) = 0;
Tinvs(1,0) = 0;
Tinvs(1,1) = prec;
arma::mat discrimvcovmat(2,2);
arma::mat discrimmumat(2,1); 
arma::mat discrimcholmat(2,2);
arma::mat abmat(2,1);
Rcpp::NumericVector IRTtempAB(2);

arma::mat Bs(nvotes,ntopics);
arma::mat Ws(nvotes,1);
arma::mat V(ntopics,ntopics);
double rho = 0.99;
V.fill(rho);
for (int k = 0; k < ntopics; k++) V(k,k) = 1;
arma::mat Vinvs = inv(V);
arma::mat idealmumat(ntopics,1);
arma::mat idealvcovmat(ntopics,ntopics);
arma::mat idealcholmat(ntopics,ntopics);
arma::mat thetamat(ntopics,1);
Rcpp::NumericVector IRTtempTheta(ntopics);

double proposalprec = 101;
double rhoproposal = 0.99;
double metropolisratio = 1;
double forwardjumplogdensity = 1;
double backwardjumplogdensity = 1;
double llcurrent = 1;
double llproposal = 1;
double burnacceptrate = 0;
double mcmcacceptrate = 0;
arma::mat llkerneltemp(1,1);
Rcpp::IntegerVector BurnAccepts(burn);
Rcpp::IntegerVector MCMCAccepts(mcmc);

double IRTll = 0;
Rcpp::NumericVector IRTllterm(1);
Rcpp::NumericVector zerovec(1);
zerovec(0) = 0;

// INITIALIZE TOPIC MODEL

Rprintf("Initializing Topic Model...\\n");  

Rcpp::NumericVector pvec = rep(1.0/ntopics,ntopics);
int wordcounter = 0;
for (int dtc = 0; dtc < ndocterms; dtc++){  // loop through each sparse matrix entry...
for (int l = 0; l < WMC(dtc); l++){  // for all appearances of the term...
LDAzmn(wordcounter) = rcat(pvec); 
LDAnmz(WMD(dtc),LDAzmn(wordcounter))++; 
LDAnm(WMD(dtc))++; 
LDAnzt(LDAzmn(wordcounter),WMT(dtc))++; 
LDAnz(LDAzmn(wordcounter))++;
wordcounter++;
}
}			

Rprintf("Initializing Ideal Point Model...\\n");

// INITIALIZE IDEAL POINT MODEL

for (int i = 0; i < nvoters; i++){ // for all voters...
for (int k = 0; k < ntopics; k++){ // for all topics...
IRTtheta(i,k) = Rf_rnorm(0,1);
}
}

for (int m = 0; m < nvotes; m++){ // for all votes...
IRTbeta(m,0) = Rf_rnorm(0,1);
IRTbeta(m,1) = Rf_rnorm(0,1); 
}


Rprintf("Initiating Markov Chain Monte Carlo...\\n");

// BEGIN MARKOV CHAIN MONTE CARLO

for (int iter = 0; iter < burn + mcmc; iter ++){

if (iter % 1 == 0) Rprintf("Beginning iteration %d of %d... \\n",iter,burn+mcmc); // print simulation status

// Reassign	each term appearance to a topic

LDAll = 0;
wordcounter = 0;
for (int dtc = 0; dtc < ndocterms; dtc++){

for (int l = 0; l < WMC(dtc); l++){  // draw topic assignment for each appearances of the term...

// unassign the current appearance of the term...
LDAnmz(WMD(dtc),LDAzmn(wordcounter))--; 
LDAnm(WMD(dtc))--; 
LDAnzt(LDAzmn(wordcounter),WMT(dtc))--; 
LDAnz(LDAzmn(wordcounter))--;

for (int k = 0; k < ntopics; k++){
pvec(k) = (LDAnmz(WMD(dtc),k) + alpha)*(LDAnzt(k,WMT(dtc)) + beta)/(LDAnz(k) + nterms*beta);
}   

// reassign the current appearance of the term...								
LDAzmn(wordcounter) = rcat(pvec); 

// save the current conditional log-likelihood of the term...	
// This yields an approximation to the final iteration log-likelihood, 
// but avoids doing all these loops twice for each iteration						
LDAll = LDAll + log(pvec(LDAzmn(wordcounter))/sum(pvec)); 

LDAnmz(WMD(dtc),LDAzmn(wordcounter))++; 
LDAnm(WMD(dtc))++; 
LDAnzt(LDAzmn(wordcounter),WMT(dtc))++; 
LDAnz(LDAzmn(wordcounter))++;
wordcounter++;
}
}


// Calculate word mixtures for each topic

for (int k = 0; k < ntopics; k++){
for (int t = 0; t < nterms; t++){
LDAphi(k,t) = (LDAnzt(k,t) + beta)/(LDAnz(k) + nterms*beta);
}
}


// Caculate topic mixtures for each document

for (int m = 0; m < nvotes; m++){
for (int k = 0; k < ntopics; k++){
LDAIRTlambda(m,k) = (LDAnmz(m,k) + alpha)/(LDAnm(m) + ntopics*alpha);
}
}


// Draw truncated random normal for each vote for each voter

for (int m = 0; m < nvotes; m++){ // for all votes...		
for (int i = 0; i < nvoters; i++){ // for all voters...
muim = -1.0*IRTbeta(m,0);
for (int k=0; k < ntopics; k++) muim = muim + IRTbeta(m,1)*LDAIRTlambda(m,k)*IRTtheta(i,k);
if (Y(m,i) == 1){ // yes vote observed
IRTystar(m,i) = -rnegtruncnorm(-muim,1.0);
}
if (Y(m,i) == 0){ // no vote observed
IRTystar(m,i) = rnegtruncnorm(muim,1.0);
}	
if (Y(m,i) == -9){ // vote missing
IRTystar(m,i) = Rf_rnorm(muim,1);
}	
}
}


// Draw normal for two discrimination parameters for each vote

for (int m = 0; m < nvotes; m++){ // for all votes...	
for (int i = 0; i < nvoters; i++){ // for all voters...			
Xs(i,0) = -1;
Xs(i,1) = 0;
for (int k=0; k < ntopics; k++) Xs(i,1) = Xs(i,1) + LDAIRTlambda(m,k)*IRTtheta(i,k);
Ys(i,0) = IRTystar(m,i);
}
discrimvcovmat = inv_sympd(trans(Xs)*Xs + Tinvs);
discrimmumat = discrimvcovmat*trans(Xs)*Ys;
discrimcholmat = chol(discrimvcovmat);
IRTtempAB = rnorm(2);
for (int el = 0; el < 2; el++) abmat(el,0) = IRTtempAB(el);
abmat = discrimmumat + discrimcholmat*abmat;
for (int el = 0; el < 2; el++) IRTbeta(m,el) = abmat(el,0);
}

// Draw normal for each topic dimensions ideal points for each voter

for (int m = 0; m < nvotes; m++){
for (int k = 0; k < ntopics; k++){
Bs(m,k) = LDAIRTlambda(m,k)*IRTbeta(m,1);	
}	
}		
for (int i = 0; i < nvoters; i++){ // for all voters...	
for (int m = 0; m < nvotes; m++){
Ws(m,0) = IRTystar(m,i) + IRTbeta(m,0);
}
idealvcovmat = inv_sympd(trans(Bs)*Bs + Vinvs);
idealmumat = idealvcovmat*trans(Bs)*Ws;
idealcholmat = chol(idealvcovmat);
IRTtempTheta = rnorm(ntopics);
for (int k = 0; k < ntopics; k++) thetamat(k,0) = IRTtempTheta(k);
thetamat = idealmumat + idealcholmat*thetamat;
for (int k = 0; k < ntopics; k++) {
IRTtheta(i,k) = thetamat(k,0);
IRTthetaarma(i,k) = IRTtheta(i,k);	
}		
}

// Draw new value of correlation between ideal point dimensions via adaptive Metropolis step with beta proposal distribution
// Note: The initial value of rho = 0.99 is held for the first 500 iterations in order to ensure that the 
// dimensions are correctly oriented in datasets where the topic model assigns the documents decisively. 
// Adaptation begins at iteration 750 and continues until the end of burn in.

if (iter > 499){

rhoproposal = Rcpp::as<double>(rbeta(1,proposalprec*rho,proposalprec*(1-rho)));

llcurrent = 0; 
for (int i = 0; i < nvoters; i++){ // for all voters...	
llkerneltemp = IRTthetaarma.row(i)*Vinvs*trans(IRTthetaarma.row(i));
llcurrent = llcurrent - (1.0/2.0)*log(det(V)) - (1.0/2.0)*llkerneltemp(0,0);
}

V.fill(rhoproposal);
for (int k = 0; k < ntopics; k++) V(k,k) = 1.0;
Vinvs = inv(V);

llproposal = 0;  
for (int i = 0; i < nvoters; i++){ // for all voters...	
llkerneltemp = IRTthetaarma.row(i)*Vinvs*trans(IRTthetaarma.row(i));
llproposal = llproposal - (1.0/2.0)*log(det(V)) - (1.0/2.0)*llkerneltemp(0,0);
}

forwardjumplogdensity = Rf_lgammafn(proposalprec) - Rf_lgammafn(proposalprec*rho) - Rf_lgammafn(proposalprec*(1-rho)) + (proposalprec*rho - 1.0)*log(rhoproposal) + (proposalprec*(1-rho) - 1.0)*log(1-rhoproposal);
backwardjumplogdensity = Rf_lgammafn(proposalprec) - Rf_lgammafn(proposalprec*rhoproposal) - Rf_lgammafn(proposalprec*(1-rhoproposal)) + (proposalprec*rhoproposal - 1.0)*log(rho) + (proposalprec*(1-rhoproposal) - 1.0)*log(1-rho);

metropolisratio = exp(llproposal + backwardjumplogdensity - llcurrent - forwardjumplogdensity + log(2*rhoproposal) - log(2*rho)); // final two terms are beta(2,1) prior on rho

if (iter < burn){
if (Rf_runif(0,1) < metropolisratio){
BurnAccepts(iter) = 1;
rho = rhoproposal;
} else {
BurnAccepts(iter) = 0;
}

if (iter > 749){
burnacceptrate = 0;
for (int iterate = iter-100; iterate < iter; iterate++) burnacceptrate = burnacceptrate + double(BurnAccepts(iterate))/100.0;
if (burnacceptrate > 0.44){
proposalprec = proposalprec*0.98;
} else {
proposalprec = proposalprec*1.02;
}
}	

} else {
if (Rf_runif(0,1) < metropolisratio){
MCMCAccepts(iter - burn) = 1;
rho = rhoproposal;
} else {
MCMCAccepts(iter - burn) = 0;
}			
}

V.fill(rho);
for (int k = 0; k < ntopics; k++) V(k,k) = 1;
Vinvs = inv(V);

}

// Compute log-likelihood for the voting data

IRTll = 0;
for (int m = 0; m < nvotes; m++){ // for all votes...		
for (int i = 0; i < nvoters; i++){ // for all voters...
muim = -1.0*IRTbeta(m,0);
for (int k=0; k < ntopics; k++) muim = muim + IRTbeta(m,1)*LDAIRTlambda(m,k)*IRTtheta(i,k);
if (Y(m,i) == 1){ // yes vote observed
IRTllterm = pnorm(zerovec,-muim,1.0);
IRTll = IRTll + log(IRTllterm(0));
}
if (Y(m,i) == 0){ // no vote observed
IRTllterm = pnorm(zerovec,muim,1.0);
IRTll = IRTll + log(IRTllterm(0));
}		
}
}



if (iter >= burn){

if (store[0]){
for (int t = 0; t < nterms; t++){
for (int k = 0; k < ntopics; k++){
wordchain(wordchainstorageloc) = LDAphi(k,t);
wordchainstorageloc++;
}
}	
} else {
for (int t = 0; t < nterms; t++){
for (int k = 0; k < ntopics; k++){
wordchain(wordchainstorageloc) = wordchain(wordchainstorageloc) + LDAphi(k,t)/mcmc;
wordchainstorageloc++;
}
}
wordchainstorageloc = 0;
}

if (store[1]){
for (int m = 0; m < nvotes; m++){
for (int k = 0; k < ntopics; k++){
topicchain(topicchainstorageloc) = LDAIRTlambda(m,k);
topicchainstorageloc++;
}
}	
} else {
for (int m = 0; m < nvotes; m++){
for (int k = 0; k < ntopics; k++){
topicchain(topicchainstorageloc) = topicchain(topicchainstorageloc) + LDAIRTlambda(m,k)/mcmc;
topicchainstorageloc++;
}
}
topicchainstorageloc = 0;
}

if (store[2]){
for (int i = 0; i < nvoters; i++){
for (int k = 0; k < ntopics; k++){
idealchain(idealchainstorageloc) = IRTtheta(i,k);
idealchainstorageloc++;
}
}	
} else {
for (int i = 0; i < nvoters; i++){
for (int k = 0; k < ntopics; k++){
idealchain(idealchainstorageloc) = idealchain(idealchainstorageloc) + IRTtheta(i,k)/mcmc;
idealchainstorageloc++;
}
}
idealchainstorageloc = 0;
}

if (store[3]){
for (int j = 0; j < 2; j++){
for (int m = 0; m < nvotes; m++){			
discrimchain(discrimchainstorageloc) = IRTbeta(m,j);
discrimchainstorageloc++;
}
}	
} else {
for (int j = 0; j < 2; j++){
for (int m = 0; m < nvotes; m++){
discrimchain(discrimchainstorageloc) = discrimchain(discrimchainstorageloc) + IRTbeta(m,j)/mcmc;
discrimchainstorageloc++;
}
}
discrimchainstorageloc = 0;
}

rhochain(iter - burn) = rho;		

llldachain(iter - burn) = LDAll;
llirtchain(iter - burn) = IRTll;

} 

} // END MARKOV CHAIN MONTE CARLO

Rprintf("Markov Chain Monte Carlo Completed...\\n");

// Load Mean Posterior Estimates into state variables for IRT model

topicchainstorageloc = 0;
if (store[1]){
for (int m = 0; m < nvotes; m++){
for (int k = 0; k < ntopics; k++){
LDAIRTlambda(m,k) = 0;
}
}
for (int iter = 0; iter < mcmc; iter++){
for (int m = 0; m < nvotes; m++){
for (int k = 0; k < ntopics; k++){
LDAIRTlambda(m,k) = LDAIRTlambda(m,k) + topicchain(topicchainstorageloc)/mcmc;
topicchainstorageloc++;
}
}	
}
} else {
for (int m = 0; m < nvotes; m++){
for (int k = 0; k < ntopics; k++){
LDAIRTlambda(m,k) = topicchain(topicchainstorageloc);
topicchainstorageloc++;
}
}		
}

idealchainstorageloc = 0;
if (store[2]){
for (int i = 0; i < nvoters; i++){
for (int k = 0; k < ntopics; k++){
IRTtheta(i,k) = 0;
}
}	
for (int iter = 0; iter < mcmc; iter++){	
for (int i = 0; i < nvoters; i++){
for (int k = 0; k < ntopics; k++){
IRTtheta(i,k) = IRTtheta(i,k) + idealchain(idealchainstorageloc)/mcmc;
idealchainstorageloc++;
}
}	
}	
} else {
for (int i = 0; i < nvoters; i++){
for (int k = 0; k < ntopics; k++){
IRTtheta(i,k) = idealchain(idealchainstorageloc);
idealchainstorageloc++;
}
}	
}

discrimchainstorageloc = 0;
if (store[3]){
for (int j = 0; j < 2; j++){
for (int m = 0; m < nvotes; m++){
IRTbeta(m,j) = 0;
}
}
for (int iter = 0; iter < mcmc; iter++){		
for (int j = 0; j < 2; j++){
for (int m = 0; m < nvotes; m++){
IRTbeta(m,j) = IRTbeta(m,j) + discrimchain(discrimchainstorageloc)/mcmc;
discrimchainstorageloc++;
}
}
}
} else {
for (int j = 0; j < 2; j++){
for (int m = 0; m < nvotes; m++){
IRTbeta(m,j) = discrimchain(discrimchainstorageloc);
discrimchainstorageloc++;
}
}	
}

// Calculate log-likelihood of IRT mean posterior in order to calculate the model DIC...

IRTll = 0;
for (int m = 0; m < nvotes; m++){ // for all votes...		
for (int i = 0; i < nvoters; i++){ // for all voters...
muim = -1.0*IRTbeta(m,0);
for (int k=0; k < ntopics; k++) muim = muim + IRTbeta(m,1)*LDAIRTlambda(m,k)*IRTtheta(i,k);
if (Y(m,i) == 1){ // yes vote observed
IRTllterm = pnorm(zerovec,-muim,1.0);
IRTll = IRTll + log(IRTllterm(0));
}
if (Y(m,i) == 0){ // no vote observed
IRTllterm = pnorm(zerovec,muim,1.0);
IRTll = IRTll + log(IRTllterm(0));
}		
}
}

Rcpp::NumericVector dicirt(1);	
dicirt(0) = 2.0*IRTll - 4.0*mean(llirtchain);

mcmcacceptrate = 0;
for (int iterate = 0; iterate < mcmc; iterate++) mcmcacceptrate = mcmcacceptrate + double(MCMCAccepts(iterate))/(1.0 + mcmc);	

return Rcpp::List::create(Rcpp::Named("word.chain") = wordchain, Rcpp::Named("topic.chain") = topicchain, Rcpp::Named("ideal.chain") = idealchain, Rcpp::Named("discrim.chain") = discrimchain, Rcpp::Named("rho.chain") = rhochain, Rcpp::Named("ll.lda.chain") = llldachain, Rcpp::Named("ll.irt.chain") = llirtchain, Rcpp::Named("dic.irt") = dicirt, Rcpp::Named("accept.rho") = mcmcacceptrate);

' 



# Compile C++ Function
LDAIRTcpp <- cxxfunction(signature(WordMatrixDocs = "integer",WordMatrixTerms = "integer",WordMatrixCounts = "integer", VoteMatrix = "integer", LDAtopics = "int", LDAterms = "int", LDAwords = "int", LDAdocterms = "int", LDAalpha = "double", LDAbeta = "double",IRTprec = "double",IRTpolarity = "int", SaveChains = "logical",BurntIterations = "int",SavedIterations = "int"), body= LDAIRTsrc, includes= LDAIRTinc, plugin = "RcppArmadillo")

LDAIRT <- function(WordSparseMatrix,VoteMatrix,LDA.topics=10,LDA.alpha=1/LDA.topics,LDA.beta=1,IRT.discrim.prec=0.25,IRT.polarity=1,burn.iter=100,mcmc.iter=100,save.word.chain=FALSE,save.topic.chain=FALSE,save.ideal.chain=TRUE,save.discrim.chain=FALSE){
  
  if (WordSparseMatrix$nrow != dim(VoteMatrix)[1]) stop("Word Sparse Matrix and Vote Matrix must have the same number of rows.")
  if (LDA.topics > WordSparseMatrix$nrow/5) stop("Too many topics requested, fewer than 5 votes per topic.")
  if (dim(VoteMatrix)[2] < abs(IRT.polarity)) stop("Invalid polarity for IRT, selected voter index is greater than Vote Matrix dimension.")
  
  ntopics <- LDA.topics
  nterms <- WordSparseMatrix$ncol
  nwords <- sum(WordSparseMatrix$v)
  nvotes <- dim(VoteMatrix)[1]
  nvoters <- dim(VoteMatrix)[2]
  ndocterms <- length(WordSparseMatrix$v)
  
  term.names <- WordSparseMatrix$dimnames$Terms
  vote.names <- rownames(VoteMatrix)
  if (is.null(vote.names)) vote.names <- paste("Vote",1:nvotes)
  voter.names <- colnames(VoteMatrix)
  if (is.null(voter.names)) voter.names <- paste("Voter",1:nvoters)
  topic.names <- paste("Topic",1:ntopics)
  
  SaveChains <- c(save.word.chain,save.topic.chain,save.ideal.chain,save.discrim.chain)		
  
  library(Rcpp)
  library(RcppArmadillo)	
  library(inline)
  
  VoteMatrixdims <- dim(VoteMatrix)
  VoteMatrix <- as.integer(VoteMatrix)
  dim(VoteMatrix) <- VoteMatrixdims
  VoteMatrix <- replace(VoteMatrix,is.na(VoteMatrix),-9)
  
  WordMatrixDocs <- WordSparseMatrix$i - 1
  WordMatrixTerms <- WordSparseMatrix$j - 1
  WordMatrixCounts <- WordSparseMatrix$v
  
  LDAIRTcpp.out <- LDAIRTcpp(WordMatrixDocs, WordMatrixTerms, WordMatrixCounts,VoteMatrix,ntopics,nterms,nwords, ndocterms,LDA.alpha,LDA.beta,IRT.discrim.prec,IRT.polarity,c(save.word.chain, save.topic.chain, save.ideal.chain, save.discrim.chain),burn.iter,mcmc.iter)
  
  print("C++ Code Completed")
  
  # Set appropriate dimensions for chain objects
  
  dim(LDAIRTcpp.out$word.chain) <- c(ntopics,nterms,mcmc.iter^save.word.chain)
  dimnames(LDAIRTcpp.out$word.chain) <- c(list(topic.names,term.names,1:(mcmc.iter^save.word.chain)))
  dim(LDAIRTcpp.out$topic.chain) <- c(ntopics,nvotes,mcmc.iter^save.topic.chain)
  dimnames(LDAIRTcpp.out$topic.chain) <- c(list(topic.names,vote.names,1:(mcmc.iter^save.topic.chain)))
  dim(LDAIRTcpp.out$ideal.chain) <- c(ntopics,nvoters,mcmc.iter^save.ideal.chain)
  dimnames(LDAIRTcpp.out$ideal.chain) <- c(list(topic.names,voter.names,1:(mcmc.iter^save.ideal.chain)))
  dim(LDAIRTcpp.out$discrim.chain) <- c(nvotes,2,mcmc.iter^save.discrim.chain)
  dimnames(LDAIRTcpp.out$discrim.chain) <- c(list(vote.names,c("IRTalpha","IRTbeta"),1:(mcmc.iter^save.discrim.chain)))
  dim(LDAIRTcpp.out$rho.chain) <- mcmc.iter
  dim(LDAIRTcpp.out$ll.lda.chain) <- mcmc.iter
  dim(LDAIRTcpp.out$ll.irt.chain) <- mcmc.iter
  dim(LDAIRTcpp.out$dic.irt) <- 1
  
  print(paste("IRT Deviance Information Criterion:",round(LDAIRTcpp.out$dic.irt,2)))
  
  gewekep.lda <- t.test(LDAIRTcpp.out$ll.lda.chain[1:(trunc(mcmc.iter/4))],LDAIRTcpp.out$ll.lda.chain[(trunc(3*mcmc.iter/4)):mcmc.iter])$p.value
  gewekep.irt <- t.test(LDAIRTcpp.out$ll.irt.chain[1:(trunc(mcmc.iter/4))],LDAIRTcpp.out$ll.irt.chain[(trunc(3*mcmc.iter/4)):mcmc.iter])$p.value
  
  print(paste("Geweke p-value for equal mean LDA log-likelihood in first and last quarter of simulation:",round(gewekep.lda,6)))
  print(paste("Geweke p-value for equal mean IRT log-likelihood in first and last quarter of simulation:",round(gewekep.irt,6)))
  
  return(LDAIRTcpp.out)
  
}




summary.LDAIRT <- function(LDAIRT.out){
  
  ntopics <- dim(LDAIRT.out$word.chain)[1]
  nterms <- dim(LDAIRT.out$word.chain)[2]
  nvotes <- dim(LDAIRT.out$discrim.chain)[1]
  nvoters <- dim(LDAIRT.out$ideal.chain)[2]
  
  # Calculate relative evidence provided by each word with respect to topic
  if (dim(LDAIRT.out$word.chain)[3] > 1) phi.est <- apply(LDAIRT.out$word.chain,c(1,2),mean) else phi.est <- LDAIRT.out$word.chain[,,1]
  wordtopicpropensity <- phi.est/t(matrix(rep(colSums(phi.est),ntopics), nterms ,ntopics))
  wordtopicevidence <- phi.est*wordtopicpropensity
  
  # Calculate voter orderings in each topic
  if (dim(LDAIRT.out$ideal.chain)[3] > 1) theta.est <- apply(LDAIRT.out$ideal.chain,c(1,2),mean) else theta.est <- LDAIRT.out$ideal.chain[,,1]
  rank.est <- t(apply(theta.est,1,rank))
  
  # Find votes most heavily in each topic
  if (dim(LDAIRT.out$topic.chain)[3] > 1) lambda.est <- apply(LDAIRT.out$topic.chain,c(1,2),mean) else lambda.est <- LDAIRT.out$topic.chain[,,1]
  
  TopWords <- matrix(NA,20,ntopics)
  TopVotes <- matrix(NA,20,ntopics)
  
  for (k in 1:ntopics){
    print(paste("Topic ",k,":",sep=""))
    TopWords[,k] <- sort(wordtopicevidence[k,],decreasing=TRUE,index.return=TRUE)$ix[1:20]
    print(paste("Distinctive Words: ",paste(names(sort(wordtopicevidence[k,],decreasing=TRUE))[1:10],collapse=", "),sep=""))
    TopVotes[,k] <- sort(lambda.est[k,],decreasing=TRUE,index.return=TRUE)$ix[1:20]
    print(paste("Most Topical Votes: ",paste(paste(colnames(lambda.est)[sort(lambda.est[k,],decreasing=TRUE,index.return=TRUE)$ix],paste("(",round(lambda.est[k,],2),")",sep="")[sort(lambda.est[k,],decreasing=TRUE,index.return=TRUE)$ix])[1:10],collapse="; "),sep=""))
    print(paste("Voter Ordering: ",paste(names(sort(rank.est[k,],decreasing=FALSE)),collapse=", "),sep=""))
    # print(paste("Locations: ",paste(sort(theta.est[k,],decreasing=FALSE),collapse=", "),sep=""))
    print("")
  }
  
  
  
  return(list(theta.est=theta.est,lambda.est=lambda.est,phi.est=phi.est,TopWordsByTopic=TopWords,TopVotesByTopic=TopVotes,Ntopics=ntopics,Nterms=nterms,Nvotes=nvotes,Nvoters=nvoters))
  
}

