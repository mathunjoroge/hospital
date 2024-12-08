<?php
session_start();
include('../connect.php');
$a = $_POST['name'];
$b = $_POST['contact'];
$c= date("Y-m-d", strtotime($_POST['age']));
$d = $_POST['nok'];
$e= $_POST['sex']; 
$f = $_POST['nokc']; 
$g= $_POST['address'];   
$h = $_POST['number'];
$dept = $_POST['dept']; 
$payment_mode = $_POST['payment_mode'];
$sql = "INSERT INTO patients (name,contact,age,next_of_kin,sex,nokcontact,address,opno,type) VALUES (:a,:b,:c,:d,:e,:f,:g,:h,:payment_mode)";
$q = $db->prepare($sql);
$q->execute(array(':a'=>$a,':b'=>$b,':c'=>$c,':d'=>$d,':e'=>$e,':f'=>$f,':g'=>$g,':h'=>$h,':payment_mode'=>$payment_mode)); 
//record to visits table
$sql = "INSERT INTO visits (patient) VALUES (:h)";
$q = $db->prepare($sql);
$q->execute(array(':h'=>$h)); 
?>
<?php
$fees = $_POST['fees'];
$j = $_POST['number'];
$doctor =$_SESSION['SESS_FIRST_NAME'];
$date=date('Y-m-d');
$List = implode(', ',$fees);

$result = $db->prepare("SELECT GROUP_CONCAT(amount) AS amount FROM  fees   WHERE fees_id IN($List)");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
 $feeamount=$row['amount'];

$amounts = explode (",", $feeamount); 


foreach (array_combine($fees, $amounts) as $fee => $amount){

$sql = "INSERT INTO collection (fees_id,date,paid_by,amount) VALUES (:a,:b,:c,:d)";
$q = $db->prepare($sql);
$q->execute(array(':a'=>$fee,':b'=>$date,':c'=>$h,':d'=>$amount));

}
}
$served=1;
$has_bill=1;
$sql ="UPDATE patients
      SET   has_bill=?,
      served=?
      WHERE opno=?";
$q = $db->prepare($sql);
$q->execute(array($has_bill,$has_bill,$h));
//if inpatient go to admit
if ($dept==2) {
header("location: admit.php?search==$h&response=0");
}

else{
header("location: receipt.php?name=$a&number=$h");
}
?>
