sumH = 0;
sumI = 0;
%upologizoume ton ari8mo twn ill kai healthy
for i=1:height(IndianLiverPatientDatasetILPD)
    if IndianLiverPatientDatasetILPD{i,11}==2
        sumH=sumH+1;
    elseif IndianLiverPatientDatasetILPD{i,11}==1
        sumI= sumI +1;
    end
end

%antika8istoume ta Male me 0 kai ta Female me 1
keys = categorical({'Male','Female'});%dhmiourgia pinaka tupou categorical me stoixeia Male kai Female
values = [0, 1];
[found, where] = ismember(IndianLiverPatientDatasetILPD{:,2}, keys);%found exei logikh timh 1 stis 8eseis opou uparxei h male h female
%to where exei times 1 h 2 analoga me to an h 8esh auth exei thn timh male
%h female antistoixa
correspondingvalues = nan(size(IndianLiverPatientDatasetILPD{:,2}));%arxikopoiei ton pinaka corresponding values
correspondingvalues(found) = values(where(found));%bazei thn 0 h 1 ston corresponding values opou exoume 1 h 2 dld male h female
clearvars found keys values where i;
M = [ IndianLiverPatientDatasetILPD{:,1} correspondingvalues IndianLiverPatientDatasetILPD{:,3:11} ];%dhmiourgoume tonkainourio pinaka me tis antikatastaseis ths deuterhs sthlhs
M(isnan(M))= nanmean(M(:,10));%antika8istoume tis NaN times me to M.O ths dekaths sthlhs xwris na upologizoume tis NaN times
N=[normalize(M(:,1:10)) M(:,11)];%kanonikopoioume ton pinaka ektos ths 11 sthlhs
clearvars correspondingvalues M;

Mdl = fitcnb(N(:,1:10),N(:,11));%ekpaideuoume ton naive bayes taxinomhth 
cl=crossval(Mdl,'Kfold',5);%ekpaideuoume ton naive bayes me 5-fold cross validation
Label = kfoldPredict(cl);%bazoume ston label tis problepseis tou taxinomhth
C = confusionmat(N(:,11),Label);%ftiaxnoume ton confusion matrix

sensitivity = C(1,1)/(C(1,1) + C(1,2));%upologizoume to sensitivity
specificity = C(2,2)/(C(2,1) + C(2,2));%upologizoume to specificity
geomM = sqrt(sensitivity*specificity); %upologizoume to geometriko meso


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cost=[0,5;1,0];
LosC=0;
GeoC=0;
Constraint=1;
sens1=0;
spec1=0;
for i=1:5:200
    SVMModel = fitcsvm(N(:,1:10),N(:,11),'BoxConstraint',i,'Cost',cost,'ClassNames',[2,1] );
    SVM = crossval(SVMModel,'KFold',5);
    Label1 = kfoldPredict(SVM);%bazoume ston label tis problepseis tou taxinomhth
    C1 = confusionmat(N(:,11),Label1);%ftiaxnoume ton confusion matrix

    sensitivity_1 = C1(1,1)/(C1(1,1) + C1(1,2));%upologizoume to sensitivity
    specificity_1 = C1(2,2)/(C1(2,1) + C1(2,2));%upologizoume to specificity
    geom_Con = sqrt(sensitivity_1*specificity_1); %upologizoume to geometriko meso

    if geom_Con>=GeoC
        LosC=kfoldLoss(SVM);
        GeoC=geom_Con;
        sens1=sensitivity_1;
        spec1=specificity_1;
        Constraint=i;
    end
end
LosG=100;
gamma=0;
GeoG=0;
sens2=0;
spec2=0;
for j=0.5:0.5:10
   SVMModel = fitcsvm(N(:,1:10),N(:,11),'KernelFunction','rbf','BoxConstraint',Constraint,'KernelScale',j,'Cost',cost,'ClassNames',[2,1] );
    SVM = crossval(SVMModel,'KFold',5);
    Label2 = kfoldPredict(SVM);%bazoume ston label tis problepseis tou taxinomhth
    C2 = confusionmat(N(:,11),Label2);%ftiaxnoume ton confusion matrix

    sensitivity_2 = C2(1,1)/(C2(1,1) + C2(1,2));%upologizoume to sensitivity
    specificity_2 = C2(2,2)/(C2(2,1) + C2(2,2));%upologizoume to specificity
    geom_G = sqrt(sensitivity_2*specificity_2); %upologizoume to geometriko meso

    if geom_G>=GeoG
        LosG=kfoldLoss(SVM);
        GeoG=geom_G;
        sens2=sensitivity_2;
        spec2=specificity_2;
        gamma=j;
    end
end  
    
    kappa=0;
    GeoKnn=0;
    sens3=0;
    spec3=0;
for k=3:15
    knnMdl=fitcknn(N(:,1:10),N(:,11),'NumNeighbors',k);
    knn=crossval(knnMdl,'KFold', 5);
    Label3 = kfoldPredict(knn);%bazoume ston label tis problepseis tou taxinomhth
    C3 = confusionmat(N(:,11),Label3);%ftiaxnoume ton confusion matrix

    sensitivity_3 = C3(1,1)/(C3(1,1) + C3(1,2));%upologizoume to sensitivity
    specificity_3 = C3(2,2)/(C3(2,1) + C3(2,2));%upologizoume to specificity
    geom_knn = sqrt(sensitivity_3*specificity_3); %upologizoume to geometriko meso
    if geom_knn>=GeoKnn
       Losknn=kfoldLoss(knn);
       GeoKnn=geom_knn;
       sens3=sensitivity_3;
       spec3=specificity_3;
       kappa=k;
    end
end
    
    clearvars Label1 Label2 Label3 k j i geom_Con geom_G geom_knn 
    
    
 
        rho=corr(N(:,1:10),N(:,11));%upologizoume tis grammikes susxetiseis pearson
              
LosN=0;
GeoN=0;
sensN=0;
specN=0;
ConstraintN=1;
for i=1:5:200
    SVMModelN = fitcsvm(N(:,[2 8:10]),N(:,11), 'BoxConstraint',i,'Cost',cost,'ClassNames',[2,1] );
    SVMN = crossval(SVMModelN,'KFold',5);
    LabelN = kfoldPredict(SVMN);%bazoume ston label tis problepseis tou taxinomhth
    CN = confusionmat(N(:,11),LabelN);%ftiaxnoume ton confusion matrix

    sensitivity_N = CN(1,1)/(CN(1,1) + CN(1,2));%upologizoume to sensitivity
    specificity_N = CN(2,2)/(CN(2,1) + CN(2,2));%upologizoume to specificity
    geom_N = sqrt(sensitivity_N*specificity_N); %upologizoume to geometriko meso

    if geom_N>=GeoN
        LosN=kfoldLoss(SVMN);
        GeoN=geom_N;
        sensN=sensitivity_N;
        specN=specificity_N;
        ConstraintN=i;
    end
end
LosN1=100;
gammaN1=0;
sensN1=0;
specN1=0;
GeoN1=0;
for j=0.5:0.5:10
   SVMModelN = fitcsvm(N(:,[2 8:10]),N(:,11),'KernelFunction','rbf','BoxConstraint',ConstraintN,'KernelScale',j,'Cost',cost,'ClassNames',[2,1] );
    SVMN = crossval(SVMModelN,'KFold',5);
    LabelN1 = kfoldPredict(SVMN);%bazoume ston label tis problepseis tou taxinomhth
    CN1 = confusionmat(N(:,11),LabelN1);%ftiaxnoume ton confusion matrix

    sensitivity_N1 = CN1(1,1)/(CN1(1,1) + CN1(1,2));%upologizoume to sensitivity
    specificity_N1 = CN1(2,2)/(CN1(2,1) + CN1(2,2));%upologizoume to specificity
    geom_N1 = sqrt(sensitivity_N1*specificity_N1); %upologizoume to geometriko meso

    if geom_N1>=GeoN1
        LosN1=kfoldLoss(SVMN);
        GeoN1=geom_N1;
        sensN1=sensitivity_N1;
        specN1=specificity_N1;
        gammaN1=j;
    end
end  
    
    clearvars sensitivity_1 specificity_1 sensitivity_2 specificity_2 sensitivity_3 specificity_3 sensitivity_N specificity_N sensitivity_N1 specificity_N1
    clearvars geom_N geom_N1 
    
    Male = [];
    Female = [];
    for i=1:583
        if N(i,2)<0
            Male=[Male(:,:);N(i,:)];
        else 
            Female=[Female(:,:);N(i,:)];
        end
    end
            
    SVMModelM = fitcsvm(Male(:,1:10),Male(:,11),'Cost',cost,'ClassNames',[2,1] );        
    SVMModelF = fitcsvm(Female(:,1:10),Female(:,11),'Cost',cost,'ClassNames',[2,1] );        
    SVMM = crossval(SVMModelM,'KFold',5);
    SVMF = crossval(SVMModelF,'KFold',5);
    LabelM = kfoldPredict(SVMM);%bazoume ston label tis problepseis tou taxinomhth
    CM = confusionmat(Male(:,11),LabelM);%ftiaxnoume ton confusion matrix

    sensitivity_M = CM(1,1)/(CM(1,1) + CM(1,2));%upologizoume to sensitivity
    specificity_M = CM(2,2)/(CM(2,1) + CM(2,2));%upologizoume to specificity
    geom_M = sqrt(sensitivity_M*specificity_M); %upologizoume to geometriko meso
    LabelF = kfoldPredict(SVMF);%bazoume ston label tis problepseis tou taxinomhth
    CF = confusionmat(Female(:,11),LabelF);%ftiaxnoume ton confusion matrix

    sensitivity_F = CF(1,1)/(CF(1,1) + CF(1,2));%upologizoume to sensitivity
    specificity_F = CF(2,2)/(CF(2,1) + CF(2,2));%upologizoume to specificity
    geom_F = sqrt(sensitivity_F*specificity_F); %upologizoume to geometriko meso
  
    sumHM = 0;
    sumIM = 0;
    %upologizoume ton ari8mo twn ill kai healthy
    for i=1:441
        if Male(i,11)==2
            sumHM = sumHM + 1;
        elseif Male(i,11)==1
            sumIM = sumIM + 1;
        end
    end

    sumHF = 0;
    sumIF = 0;
    for i=1:142
        if Female(i,11)==2
            sumHF = sumHF + 1;
        elseif Female(i,11)==1
            sumIF = sumIF + 1;
        end
    end
    
    
    
    